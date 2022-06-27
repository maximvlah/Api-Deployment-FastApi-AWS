import argparse
import base64
import io
import os
import logging
import sys

import yaml
import boto3
import base64
import pickle

import os
import json
import base64
from io import BytesIO
import requests

from annoy import AnnoyIndex

from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.efficientnet import preprocess_input

from urllib.parse import urlparse

from aiohttp.client import ClientSession
from asyncio import wait_for, gather, Semaphore

from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel, validator

import numpy as np

from PIL import Image

from mangum import Mangum


THREAD_COUNT = int(os.environ.get('THREAD_COUNT', 5))
"""The number of threads used to download and process image content."""

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
"""The number of images to process in a batch."""

TIMEOUT = int(os.environ.get('TIMEOUT', 30))
"""The timeout to use when downloading files."""


logger = logging.getLogger(__name__)


class HealthCheck(BaseModel):
    """
    Represents an image to be predicted.
    """
    message: Optional[str] = 'OK'


class ImageInput(BaseModel):
    """
    Represents an image to be predicted.
    """
    url: Optional[str] = None
    data: Optional[str] = None


class ImageOutput(BaseModel):
    """
    Represents the result of a prediction
    """
    embedding: List[float] = None

class PredictRequest(BaseModel):
    """
    Represents a request to process
    """
    images: List[ImageInput] = []

class PredictResponse(BaseModel):
    """
    Represents a request to process
    """
    embeddings: List[ImageOutput] = []

class ProductSearchResponse(BaseModel):
    """
    Represents a request to process
    """
    best_matches: dict = None

class ImageNotDownloadedException(Exception):
    pass

def download_files_from_s3(bucket_name,
                           s3_client,
                           files):
    """Downloads files from S3 bucket
    Args:
        bucket_name (str): name of S3 bucket
        s3_client (boto3.client): S3 client
        files (list[str]): list of files to download
    """
    # Get list of files in S3 bucket
    for f in tqdm(files, desc="Downloading files from S3", total=len(files)):

        if not os.path.exists(f):
            s3_client.download_file(bucket_name, f, f)
        else:
            print(f"{f} already exists!")
    
def retireve_data_from_aws(idxs,s3_client,table,partition_key="code"):
    """Retrieves data for product from AWS
    Args:
        idxs (list[int]): best match idxs
        s3_client (boto3.client): S3 client
        table (boto3.dynamodb.Table): DynamoDB table
    
    Returns:
        data (dict): data for best matches
    """

    search_results = {}
    for idx in range(len(idxs)):

        # Get data from DynamoDB
        ean = IDX_TO_EAN_MAP[idxs[idx]]
        response = table.get_item(
            Key={
                partition_key: str(ean) #NOTE: converting to string to match DynamoDB partition key type
            }
        )
        best_matches_metadata = response['Item']

        # Load image from S3 bucket and convert to Base64
        #NOTE: using crops of original images instead of original images
        s3_image = s3_client.get_object(Bucket=CONFIG["bucket_name"], Key=f"crops/{ean}_.jpg") 
        s3_image = Image.open(io.BytesIO(s3_image['Body'].read())).convert('RGB')
        best_matches_metadata['image_b64'] = img2b64(s3_image)

        # Append to search_results
        search_results[str(idx)] = best_matches_metadata

    return search_results

def load_annoy(path,
               num_dimensions=2048,
               metric="angular"):
    """
    Loads Annoy Index from disk

    Args:
        path (str): path to Annoy Index
        num_dimensions (int): number of dimensions of the embedding vector
        metric (str): metric to use for Annoy Index

    Returns:
        annoy_index (AnnoyIndex): Annoy Index

    """

    annoy_index = AnnoyIndex(num_dimensions,metric)
    annoy_index.load(path)

    print(f"Loaded Annoy Index from {path}")
    return annoy_index

def img2b64(image):
    """Converts PIL image to Base64 encoded string
    Args:
        image (PIL.image)
    Returns:
        b64 (str): Base64 encoded string
    """

    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str.decode("utf-8")

def b642bytes(b64_string):
    """
    Converts b64 string to bytes
    :param b64_string: string to convert
    :return: bytes
    """
    bytes = str.encode(b64_string)
    bytes = base64.b64decode(bytes)
    return bytes

with open("serve_config.yaml", "r") as stream:
    try:
        CONFIG = (yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
        CONFIG = None
assert CONFIG is not None, "Could not load config file."

 # FastAPI app
app = FastAPI()

# Connect to DynamoDB
dynamodb = boto3.resource('dynamodb',region_name=CONFIG["region_name"])
table = dynamodb.Table(CONFIG["table_name"])

# Connect to S3
s3_client = boto3.client('s3',region_name=CONFIG["region_name"])

# Download files needed for vector search from S3
# TODO: What if the search index will be very large?
files = ["search_index.ann","idx_to_ean_map.pickle"]
download_files_from_s3(CONFIG["bucket_name"],
                       s3_client,
                       files)

# Load mapping from idx to EAN.
with open("idx_to_ean_map.pickle", "rb") as f: 
    IDX_TO_EAN_MAP = pickle.load(f)

# Load Annoy Search Index
annoy = load_annoy("search_index.ann", CONFIG["embedding_size"], CONFIG["metric"])


@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for all other errors.
    """
    return JSONResponse(status_code=500, content={'message': 'Internal error.'})


@app.exception_handler(ImageNotDownloadedException)
async def client_exception_handler(request: Request, exc: ImageNotDownloadedException):
    """
    Called when the image could not be downloaded.
    """
    return JSONResponse(status_code=400, content={'message': 'One or more images could not be downloaded.'})


@app.on_event('startup')
def load_model():
    """
    Loads the model prior to the first request.
    """
    if not hasattr(app.state, 'model'):
        configure_logging()
        logger.info('Loading models...')
        app.state.model = ImageEmbedder()


def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the application.
    """
    root = logging.getLogger()
    root.handlers.clear()
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    root.setLevel(logging_level)
    root.addHandler(stream_handler)


# class ImageClassifier:
#     """
#     Classifies images according to ImageNet categories.
#     """
#     def __init__(self):
#         """
#         Prepares the model used by the application for use.
#         """
#         self.model = MobileNetV2()
#         _, height, width, channels = self.model.input_shape
#         self.input_width = width
#         self.input_height = height
#         self.input_channels = channels

#     def _prepare_images(self, images):
#         """
#         Prepares the images for prediction.

#         :param images: The list of images to prepare for prediction in Pillow Image format.

#         :return: A list of processed images.
#         """
#         batch = np.zeros((len(images), self.input_height, self.input_width, self.input_channels), dtype=np.float32)
#         for i, image in enumerate(images):
#             x = image.resize((self.input_width, self.input_height), Image.BILINEAR)
#             batch[i, :] = np.array(x, dtype=np.float32)
#         batch = preprocess_input(batch)
#         return batch

#     def predict(self, images, batch_size):
#         """
#         Predicts the category of each image.

#         :param images: A list of images to classify.
#         :param batch_size: The number of images to process at once.

#         :return: A list containing the predicted category and confidence score for each image.
#         """
#         batch = self._prepare_images(images)
#         scores = self.model.predict(batch, batch_size)
#         results = decode_predictions(scores, top=1)
#         return results

class ImageEmbedder:
    """
    Calculates image embedding.
    """
    def __init__(self):
        """
        Prepares the model used by the application for use.
        """
        # self.model = tf.keras.models.load_model("/home/maxim/projects/ourz/Api-Deployment-FastApi-AWS/models/embedder")
        m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",trainable=False),
        ])
        m.build([None, 224, 224, 3])

        self.model = m

        _, height, width, channels = None,224,224,3
        self.input_width = width
        self.input_height = height
        self.input_channels = channels

    def _preprocess_img(self,bytes):
        """
        Processes image bytes in format suitable for tf inference

        :param bytes (bytes)
        :return img (np.array): preprocessed image
        """
        img = tf.io.decode_image(bytes,channels=self.input_channels)

        img = tf.image.resize(img, 
                              method="bilinear", 
                              size=(self.input_width,self.input_height))
        img = tf.keras.preprocessing.image.img_to_array(img)
        return img

    def _prepare_images(self, images):
        """
        Prepares the images for prediction.

        :param images: The list of images to prepare for prediction in Pillow Image format.

        :return: A list of processed images.
        """
        batch = np.zeros((len(images), self.input_height, self.input_width, self.input_channels), dtype=np.float32)
        for i, image_bytes in enumerate(images):
            # x = image.resize((self.input_width, self.input_height), Image.BILINEAR)
            batch[i, :] = np.array(self._preprocess_img(image_bytes), dtype=np.float32)
        batch = preprocess_input(batch)
        return batch

    def predict(self, images, batch_size):
        """
        Predicts the category of each image.

        :param images: A list of images to classify.
        :param batch_size: The number of images to process at once.

        :return: A list containing the predicted category and confidence score for each image.
        """
        batch = self._prepare_images(images)
        embeddings = self.model.predict(batch, batch_size)
        return embeddings

def get_url_scheme(url, default_scheme='unknown'):
    """
    Returns the scheme of the specified URL or 'unknown' if it could not be determined.
    """
    result = urlparse(url, scheme=default_scheme)
    return result.scheme


async def retrieve_content(entry, sess, sem):
    """
    Retrieves the image content for the specified entry.
    """
    raw_data = None
    if entry.data is not None:
        # raw_data = base64.b64decode(entry.data)
        raw_data = b642bytes(entry.data)
    elif entry.url is not None:
        source_uri = entry.url
        scheme = get_url_scheme(source_uri)
        if scheme in ('http', 'https'):
            raw_data = await download(source_uri, sess, sem)
        else:
            raise ValueError('Invalid scheme: %s' % scheme)
    if raw_data is not None:
        # image = Image.open(io.BytesIO(raw_data))
        # image = image.convert('RGB')
        # return image
        return raw_data
    return None


async def retrieve_images(entries):
    """
    Retrieves the images for processing.

    :param entries: The entries to process.

    :return: The retrieved data.
    """
    tasks = list()
    sem = Semaphore(THREAD_COUNT)
    async with ClientSession() as sess:
        for entry in entries:
            tasks.append(
                wait_for(
                    retrieve_content(entry, sess, sem),
                    timeout=TIMEOUT,
                )
            )
        return await gather(*tasks)


async def download(url, sess, sem):
    """
    Downloads an image from the specified URL.

    :param url: The URL to download the image from.
    :param sess: The session to use to retrieve the data.
    :param sem: Used to limit concurrency.

    :return: The file's data.
    """
    async with sem, sess.get(url) as res:
        logger.info('Downloading %s' % url)
        content = await res.read()
        logger.info('Finished downloading %s' % url)
    if res.status != 200:
        raise ImageNotDownloadedException('Could not download image.')
    return content


def predict_images(images):
    """
    Predicts the image's category and transforms the results into the output format.

    :param images: The Pillow Images to predict.

    :return: The prediction results.
    """
    response = list()
    results = app.state.model.predict(images, BATCH_SIZE)[0]
    print(results.shape)

    # response.append(ImageOutput(embedding=results.tolist()))
    return results

@app.post('/v1/predict', response_model=ProductSearchResponse)
async def process(req: PredictRequest):
    """
    Predicts the category of the images contained in the request.

    :param req: The request object containing the image data to predict.

    :return: Search results.
    """
    logger.info('Processing request...')
    logger.debug(req.json())
    logger.info('Downloading images...')
    images = await retrieve_images(req.images)
    logger.info('Performing prediction...')
    predictions = predict_images(images)
    logger.info('Searching...')
    best_matches_idxs = annoy.get_nns_by_vector(predictions,CONFIG["k"])
    logger.info('Retrieving data from AWS...')
    search_results = retireve_data_from_aws(best_matches_idxs,s3_client,table)
    logger.info('Transaction complete.')

    return ProductSearchResponse(best_matches=search_results)

@app.get('/health')
def test():
    """
    Can be called by load balancers as a health check.
    """
    return HealthCheck()


handler = Mangum(app)

if __name__ == '__main__':

    import uvicorn

    parser = argparse.ArgumentParser(description='Runs the API locally.')
    parser.add_argument('--port',
                        help='The port to listen for requests on.',
                        type=int,
                        default=8080)
    args = parser.parse_args()
    configure_logging()
    uvicorn.run(app, host='0.0.0.0', port=args.port)