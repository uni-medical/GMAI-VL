import os
from io import BytesIO

import torch
from datasets import load_from_disk
from mmengine import fileio
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from xtuner._lite.chat import ChatMessages
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from .format import OPENAI_FORMAT_MAP
from .text import SoftPackerForText, TextTokenizedDataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = 5000000000
from io import BytesIO
# from petrel_client.client import Client
from PIL import ImageFile

from xtuner._lite import get_logger
logger = get_logger()

ImageFile.LOAD_TRUNCATED_IMAGES=True 
# client = Client("~/petreloss.conf")
from timeout_decorator import timeout

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


@timeout(30)
def read_img_general(img_path):
    if "mimic-cxr" in img_path:
        img_path = img_path.replace('.png', '.jpg')
    img_path = img_path.replace("langchao:s3://multi_modal/playground/data/", "/cpfs01/shared/gmai/medical_preprocessed/image-text/general_data/").replace("image-text/report_generation/", "report_generation/").replace("s3://", "/cpfs01/shared/gmai/").replace("/mnt/petrelfs/share_data/wangweiyun/share_data_eval/", "/cpfs01/shared/gmai/medical_preprocessed/image-text/general_data/")
    if "s3://" in img_path:
        assert 0
        img_bytes = client.get(img_path)
        if img_bytes is None:
            return None
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        image_size = image.size
        max_size = (50000, 50000)
        # 判断图片是否特别大
        if image_size[0] > max_size[0] or image_size[1] > max_size[1]:
            # 图片尺寸超过了阈值，进行调整
            target_size = (2048, 2048)  # 修改为你想要的大小
            image = image.resize(target_size)
        return image
    else:
        image_file = os.path.join(img_path)
        # if "/mnt/hwfile/chenzhe1/workspace_ltb/xtuner/" in str(image_file):
        #     image_file = image_file.replace("/mnt/hwfile/chenzhe1/workspace_ltb/xtuner/", "/mnt/petrelfs/share_data/wangweiyun/share_data_eval/chemistry_data/")
        return Image.open(image_file).convert('RGB')



class LlavaTokenizeFunction():

    def __init__(self,
                 tokenizer,
                 chat_template,
                 per_img_tokens,
                 image_dir=None,
                 raw_format='llava'):

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_dir = image_dir
        self.raw_format = raw_format
        self.per_img_tokens = per_img_tokens

    def __call__(self, item):
        try:
            formatter = OPENAI_FORMAT_MAP[self.raw_format]
            msg = ChatMessages.from_dict(formatter(item))
            tokenized = msg.tokenize(self.tokenizer, self.chat_template)
            tokenized['num_img_tokens'] = 0

            if 'image_urls' in tokenized:
                image_urls = tokenized['image_urls']

                image_urls = []
                for url in tokenized['image_urls']:

                    if self.image_dir:
                        image_urls.append(os.path.join(self.image_dir, url))
                    else:
                        image_urls.append(url)

                num_images = len(image_urls)
                num_img_tokens = [self.per_img_tokens for url in image_urls]
                tokenized['num_tokens'] += sum(num_img_tokens) - num_images
                tokenized['num_img_tokens'] = sum(num_img_tokens)
                tokenized['image_urls'] = image_urls

            return tokenized
        except Exception as e:
            print(e)
            print(item)
            return None



class LlavaTokenizedDataset(TextTokenizedDataset):

    def __init__(self, dataset, image_processor, max_length):
        super().__init__(dataset, max_length)
        self.image_processor = image_processor

    def process_tokenized_data(self, tokenized_data):
        images = []
        for url in tokenized_data['image_urls']:
            img = read_img_general(url)
            assert img is not None, f"read image: {url} is None"
            # img = Image.open(BytesIO(fileio.get(url)))
            images.append(img)

        if len(images):
            outputs = self.image_processor(images, return_tensors='pt')
            pixel_values = outputs['pixel_values']
        else:
            pixel_values = None

        data = {
            'input_ids': tokenized_data['input_ids'],
            'labels': tokenized_data['labels'],
            'pixel_values': pixel_values,
            'num_tokens': [tokenized_data['num_tokens']],
            'num_img_tokens': [tokenized_data['num_img_tokens']],
        }

        return data

    @classmethod
    def from_cache(cls, cache_dir, image_processor, max_length):
        dataset = load_from_disk(os.path.join(cache_dir, 'dataset'))
        ret = cls(dataset, image_processor, max_length)
        ret.cache(cache_dir)
        return ret

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        if self.cached:
            self.load_cache()
        tokenized_data = self.dataset[item]

        if self.cached:
            self._free()
        while True:
            try:
                result = self.process_tokenized_data(tokenized_data)
                return result 
            except Exception as e:
                logger.debug(f"Error {e} is happened.")
                logger.debug(f"{tokenized_data['image_urls']} error!")
                import random
                rand_idx = random.randint(0, item)
                return self.__getitem__(rand_idx)
            

class LlavaRawDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, image_processor, max_length, tokenize_fn):
        super().__init__()

        self.dataset = dataset
        self.image_processor = image_processor
        self.max_length = max_length
        self.tokenize_fn = tokenize_fn

    def process_tokenized_data(self, tokenized_data):
        images = []
        for url in tokenized_data['image_urls']:
            img = read_img_general(url)
            assert img is not None, f"read image: {url} is None"
            images.append(img)

        if len(images):
            outputs = self.image_processor(images, return_tensors='pt')
            pixel_values = outputs['pixel_values']
        else:
            pixel_values = None

        data = {
            'input_ids': tokenized_data['input_ids'],
            'labels': tokenized_data['labels'],
            'pixel_values': pixel_values,
            'num_tokens': [tokenized_data['num_tokens']],
            'num_img_tokens': [tokenized_data['num_img_tokens']],
        }

        return data

    def __getitem__(self, item):

        raw_data = self.dataset[item]
        tokenized_data = self.tokenize_fn(raw_data)
        # return self.process_tokenized_data(tokenized_data)
        while True:
            try:
                return self.process_tokenized_data(tokenized_data)
            except Exception as e:
                logger.debug(f"Error {e} is happened.")
                logger.debug(f"{tokenized_data['image_urls']} error!")
                import random
                rand_idx = random.randint(0, self.max_length)
                return self.__getitem__(rand_idx)


class SoftPackerForLlava(SoftPackerForText):

    def __init__(self,
                 dataset,
                 image_processor,
                 max_length=2048,
                 pack_info=None):
        super().__init__(dataset, max_length, pack_info)
        self.image_processor = image_processor

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        while True:
            try:
                if self.cached:
                    self.load_cache()
                dataset = self.dataset
                pack_info = self.pack_info
                packed_items = pack_info[item]['indices']
                assert len(packed_items) > 0
                packed_input_ids = []
                packed_labels = []
                packed_img_urls = []
                packed_num_tokens = []
                packed_num_img_tokens = []
                for i in packed_items:
                    packed_input_ids.extend(dataset[i]['input_ids'])
                    packed_labels.extend(dataset[i]['labels'])
                    _num_tokens = dataset[i]['num_tokens']
                    packed_num_tokens.append(_num_tokens)
                    if 'image_urls' in dataset[i]:
                        packed_img_urls.extend(dataset[i]['image_urls'])
                    if 'num_img_tokens' in dataset[i]:
                        _num_img_tokens = dataset[i]['num_img_tokens']
                        assert  _num_img_tokens is not None, logger.debug(packed_img_urls, 1)
                        packed_num_img_tokens.append(_num_img_tokens)
                images = []
                for url in packed_img_urls:
                    img = read_img_general(url)
                    assert img is not None, f"read image: {url} is None"
                    img = expand2square(
                        img,
                        tuple(
                            int(x* 255) for x in self.image_processor.image_mean))
                    # img = Image.open(BytesIO(fileio.get(url)))
                    images.append(img)
                if len(images):
                    outputs = self.image_processor(images, return_tensors='pt')
                    pixel_values = outputs['pixel_values']
                else:
                    pixel_values = None
                if sum(packed_num_tokens) < self.max_length:
                    num_pad_tokens = self.max_length - sum(packed_num_tokens)
                    packed_input_ids.extend([DEFAULT_PAD_TOKEN_INDEX] * num_pad_tokens)
                    packed_labels.extend([IGNORE_INDEX] * num_pad_tokens)
                    packed_num_tokens.append(num_pad_tokens)
                else:
                    packed_num_tokens.append(0)
                packed = {
                    'input_ids': packed_input_ids,
                    'labels': packed_labels,
                    'pixel_values': pixel_values,
                    'num_tokens': packed_num_tokens,
                    'num_img_tokens': packed_num_img_tokens
                }
                # import numpy as np
                # if pixel_values.shape[0] != np.sum(np.array(packed_input_ids) ==92550):
                #     print(pixel_values.shape[0], np.sum(np.array(packed_input_ids) ==92550), dataset[item]['image_urls'])
        
                if self.cached:
                    self._free()
                return packed
            except Exception as e:
                logger.debug(f"Error {e} is happened.")
                import random
                # rand_idx = random.randint(0, item)
                rand_idx = item - 1 
                return self.__getitem__(rand_idx)

    @classmethod
    def from_cache(cls, cache_dir, image_processor, max_length):

        dataset = load_from_disk(os.path.join(cache_dir, 'dataset'))

        pack_info_dir = os.path.join(cache_dir, f'pack-info-soft-{max_length}')
        if os.path.exists(pack_info_dir):
            pack_info = load_from_disk(pack_info_dir)
        else:
            pack_info = cls.get_pack_info(dataset, max_length)

        ret = cls(dataset, image_processor, max_length, pack_info)
        ret.cache(cache_dir)
        return ret


class LlavaCollator():

    def __init__(self, pack_batch=False):
        self.pack_batch = pack_batch

    def __call__(self, instances):

        pad_index = DEFAULT_PAD_TOKEN_INDEX

        input_ids = []
        labels = []
        attention_mask = []
        pixel_values = []
        num_tokens = []
        num_img_tokens = []

        for data in instances:
            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))
            num_tokens.extend(data['num_tokens'])
            num_img_tokens.extend(data['num_img_tokens'])
            if data['pixel_values'] is not None:
                pixel_values.append(data['pixel_values'])
            # breakpoint()
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        num_tokens = torch.IntTensor(num_tokens)
        num_img_tokens = torch.IntTensor(num_img_tokens)

        if len(instances) > 1 and self.pack_batch:

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)

        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            'num_tokens': num_tokens,
            'num_img_tokens': num_img_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict
