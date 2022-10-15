import unittest

from tqdm import tqdm

from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.parse_config import ConfigParser
import hw_asr.model as module_arch


class TestModel(unittest.TestCase):
    def test_deepspeech(self):
        config_parser = ConfigParser.get_test_configs()
        config_parser["arch"]["type"] = "KindaDeepSpeechModel"
        with clear_log_folder_after_use(config_parser):
            ds = LibrispeechDataset(
                "dev-clean", text_encoder=config_parser.get_text_encoder(),
                config_parser=config_parser
            )

            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            model = config_parser.init_obj(config_parser["arch"], module_arch, n_class=len(config_parser.get_text_encoder()))
            x = model(**batch)
            