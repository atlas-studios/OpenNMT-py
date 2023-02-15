#!/usr/bin/env python
"""REST Translation server."""
import codecs
import sys
import os
import time
import json
import threading
import re
import traceback
import importlib
import torch
import onmt.opts

from itertools import islice, zip_longest
from copy import deepcopy
from argparse import Namespace

from onmt.constants import DefaultTokens
from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed
from onmt.utils.misc import check_model_config
from onmt.utils.alignment import to_word_align
from onmt.utils.parse import ArgumentParser
from onmt.translate.translator import build_translator
from onmt.transforms.features import InferFeatsTransform
from onmt.inputters.text_utils import (textbatch_to_tensor,
                                       parse_features,
                                       append_features_to_text)
from onmt.inputters.inputter import IterOnDevice

from models import MT_Model
from database import get_db
from datetime import datetime

# Get env variables
from dotenv import load_dotenv
load_dotenv()
TIMEOUT = int(os.getenv("TIMEOUT"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
DEFAULT_MODEL = int(os.getenv("DEFAULT_MODEL"))
DEFAULT_MODEL_NAME = str(os.getenv("DEFAULT_MODEL_NAME"))
MIN_DECODING_LENGTH = int(os.getenv("MIN_DECODING_LENGTH"))
MAX_DECODING_LENGTH = int(os.getenv("MAX_DECODING_LENGTH"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE"))
LENGTH_PENALTY = float(os.getenv("LENGTH_PENALTY"))
NUM_HYPOTHESES = int(os.getenv("NUM_HYPOTHESES"))
MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE_NUMBER = int(os.getenv("DEVICE_NUMBER"))
DISABLE_UNK = bool(os.getenv("DISABLE_UNK"))
NO_REPEAT_NGRAM = int(os.getenv("NO_REPEAT_NGRAM"))


def critical(func):
    """Decorator for critical section (mutually exclusive code)"""
    def wrapper(server_model, *args, **kwargs):
        if sys.version_info[0] == 3:
            if not server_model.running_lock.acquire(True, 120):
                raise ServerModelError("Model %d running lock timeout"
                                       % server_model.model_id)
        else:
            # semaphore doesn't have a timeout arg in Python 2.7
            server_model.running_lock.acquire(True)
        try:
            o = func(server_model, *args, **kwargs)
        except (Exception, RuntimeError):
            server_model.running_lock.release()
            raise
        server_model.running_lock.release()
        return o
    return wrapper


class Timer:
    def __init__(self, start=False):
        self.stime = -1
        self.prev = -1
        self.times = {}
        if start:
            self.start()

    def start(self):
        self.stime = time.time()
        self.prev = self.stime
        self.times = {}

    def tick(self, name=None, tot=False):
        t = time.time()
        if not tot:
            elapsed = t - self.prev
        else:
            elapsed = t - self.stime
        self.prev = t

        if name is not None:
            self.times[name] = elapsed
        return elapsed


class ServerModelError(Exception):
    pass


class CTranslate2Translator(object):
    """
    This class wraps the ctranslate2.Translator object to
    reproduce the onmt.translate.translator API.
    """

    def __init__(self, model_path, target_prefix=False, preload=True):
        import ctranslate2
        try:
            self.translator = ctranslate2.Translator(model_path, "cuda", device_index=DEVICE_NUMBER)
        except:
            print(f"Error loading model; using default model {DEFAULT_MODEL_NAME}")
            self.translator = ctranslate2.Translator(MODEL_PATH+DEFAULT_MODEL_NAME, "cuda", device_index=DEVICE_NUMBER)


    @staticmethod
    def convert_onmt_to_ct2_opts(ct2_translator_args,
                                 ct2_translate_batch_args, opt):

        def setdefault_if_exists_must_match(obj, name, value):
            if name in obj:
                assert value == obj[name], f"{name} is different in"\
                    " OpenNMT-py config and in CTranslate2 config"\
                    f" ({value} vs {obj[name]})"
            else:
                obj.setdefault(name, value)

        default_for_translator = {
            "inter_threads": 1,
            "intra_threads": torch.get_num_threads(),
            "compute_type": "default",
        }
        for name, value in default_for_translator.items():
            ct2_translator_args.setdefault(name, value)

        onmt_for_translator = {
            "device": "cuda" if opt.cuda else "cpu",
            "device_index": opt.gpu if opt.cuda else 0,
        }
        for name, value in onmt_for_translator.items():
            setdefault_if_exists_must_match(
                ct2_translator_args, name, value)

        onmt_for_translate_batch_enforce = {
            "beam_size": opt.beam_size,
            "max_batch_size": opt.batch_size,
            "num_hypotheses": opt.n_best,
            "max_decoding_length": opt.max_length,
            "min_decoding_length": opt.min_length,
        }
        for name, value in onmt_for_translate_batch_enforce.items():
            setdefault_if_exists_must_match(
                ct2_translate_batch_args, name, value)

    def translate(self, batch, batch_size=8):
        preds = self.translator.translate_batch(
            batch,
            min_decoding_length = MIN_DECODING_LENGTH,
            max_decoding_length = MAX_DECODING_LENGTH,
            beam_size = BEAM_SIZE,
            replace_unknowns = True,
            length_penalty = LENGTH_PENALTY,
            num_hypotheses = NUM_HYPOTHESES)
        predictions = [[" ".join(item["tokens"]) for item in ex]
                       for ex in preds]
        return predictions

    def to_cpu(self):
        self.translator.unload_model(to_cpu=True)

    def to_gpu(self):
        self.translator.load_model()


def parse_features_opts(conf):
    features_opt = conf.get("features", None)
    if features_opt is not None:
        features_opt["n_src_feats"] = features_opt.get("n_src_feats", 0)
        features_opt["src_feats_defaults"] = \
            features_opt.get("src_feats_defaults", None)
        features_opt["reversible_tokenization"] = \
            features_opt.get("reversible_tokenization", "joiner")
    return features_opt


class TranslationServer(object):
    def __init__(self):
        self.models = {}
        self.next_id = 0

    def start(self):
        self.preload_model(model_id=str(DEFAULT_MODEL))

    def clone_model(self, model_id, opt, timeout=-1):
        """Clone a model `model_id`.

        Different options may be passed. If `opt` is None, it will use the
        same set of options
        """
        if model_id in self.models:
            if opt is None:
                opt = self.models[model_id].user_opt
            opt["models"] = self.models[model_id].opt.models
            return self.load_model(opt, timeout)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def load_model(self, opt, model_id=None, **model_kwargs):
        """Load a model given a set of options
        """
        model_id = self.preload_model(model_id=model_id, **model_kwargs)
        load_time = self.models[model_id].load_time

        return model_id, load_time

    def preload_model(self, model_id=None, **model_kwargs):
        """Preloading the model: updating internal datastructure
        It will effectively load the model if `load` is set
        """
        if model_id is not None:
            if model_id in self.models.keys():
                raise ValueError(f"Model ID %{model_id} already exists")
        else:
            model_id = self.next_id
            while model_id in self.models.keys():
                model_id += 1
            self.next_id = model_id + 1
        print(f"Pre-loading model {model_id}")
        model = ServerModel(model_id)
        self.models[model_id] = model

        return model_id

    def run(self, inputs):
        """Translate `inputs`

        We keep the same format as the Lua version i.e.
        ``[{"id": model_id, "src": "sequence to translate"},{ ...}]``

        We use inputs[0]["id"] as the model id
        """

        model_id = inputs[0].get("id", 0)
        text = [x["src"] for x in inputs]
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(text)
        else:
            print("Error No such model '%s'" % str(model_id))
            raise ServerModelError("No such model '%s'" % str(model_id))

    def unload_model(self, model_id):
        """Manually unload a model.

        It will free the memory and cancel the timer
        """

        if model_id in self.models and self.models[model_id] is not None:
            self.models[model_id].unload()
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def list_models(self):
        """Return the list of available models
        """
        models = []
        for _, model in self.models.items():
            models += [model.to_dict()]
        return models


class ServerModel(object):
    """Wrap a model with server functionality.

    Args:
        opt (dict): Options for the Translator
        model_id (int): Model ID
        preprocess_opt (list): Options for preprocess processus or None
        tokenizer_opt (dict): Options for the tokenizer or None
        postprocess_opt (list): Options for postprocess processus or None
        custom_opt (dict): Custom options, can be used within preprocess or
            postprocess, default None
        load (bool): whether to load the model during :func:`__init__()`
        timeout (int): Seconds before running :func:`do_timeout()`
            Negative values means no timeout
        on_timeout (str): Options are ["to_cpu", "unload"]. Set what to do on
            timeout (see :func:`do_timeout()`.)
        model_root (str): Path to the model directory
            it must contain the model and tokenizer file
    """

    def __init__(self, model_id, preprocess_opt=None, tokenizer_opt=None,
                 postprocess_opt=None, custom_opt=None, load=False, timeout=TIMEOUT,
                 on_timeout="unload", model_root="./", ct2_model=None,
                 features_opt=None):
        self.model_root = MODEL_PATH
        self.custom_opt = custom_opt
        model_id = str(model_id)
        temp = next(get_db()).query(MT_Model).filter(MT_Model.cbid == model_id).first()
        self.ct2_model_name = temp.name if temp else DEFAULT_MODEL_NAME
        self.model_id = int(model_id)
        self.preprocess_opt = preprocess_opt
        self.tokenizers_opt = tokenizer_opt
        self.features_opt = features_opt
        self.postprocess_opt = postprocess_opt
        self.timeout = timeout
        self.on_timeout = on_timeout

        self.ct2_model = os.path.join(self.model_root, self.ct2_model_name) \
            if self.ct2_model_name is not None else DEFAULT_MODEL_NAME

        self.unload_timer = None
        self.tokenizers = None

        self.loading_lock = threading.Event()
        self.loading_lock.set()
        self.running_lock = threading.Semaphore(value=1)

        set_random_seed(self.opt.seed, self.opt.cuda)

        if load:
            self.load(preload=True)
            self.stop_unload_timer()

    @property
    def loaded(self):
        return hasattr(self, 'translator')

    def load(self, preload=False):
        self.loading_lock.clear()

        timer = Timer()
        self.logger.info("Loading model %d" % self.model_id)
        timer.start()

        try:
            if self.ct2_model is not None:
                self.translator = CTranslate2Translator(
                    self.ct2_model,
                    preload=preload)

        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        timer.tick("model_loading")
        self.load_time = timer.tick()
        self.reset_unload_timer()
        self.loading_lock.set()

    @critical
    def run(self, inputs):
        """Translate `inputs` using this model

        Args:
            inputs (List[dict[str, str]]): [{"src": "..."},{"src": ...}]

        Returns:
            result (list): translations
            times (dict): containing times
        """

        self.stop_unload_timer()

        timer = Timer()
        timer.start()

        self.logger.info(f"Running translation {self.ct2_model_name}")

        if not self.loading_lock.is_set():
            self.logger.info(f"Model #{self.ct2_model_name} is being loaded by another thread, waiting")
            if not self.loading_lock.wait(timeout=30):
                raise ServerModelError(f"Model {self.ct2_model_name} loading timeout")

        else:
            if not self.loaded:
                self.load()
                timer.tick(name="load")
            elif self.opt.cuda:
                self.to_gpu()
                timer.tick(name="to_gpu")

        predictions = []
        texts_to_translate = []
        if len(inputs) > 0:
            for i, inp in enumerate(inputs):
                texts_to_translate.append(inp)
            try:
                predictions = self.translator.translate(
                    texts_to_translate,
                    batch_size=min(BATCH_SIZE,len(inputs)))
            except (RuntimeError, Exception) as e:
                err = "Error: %s" % str(e)
                print(err)
                print("repr(text_to_translate): "
                                  + repr(texts_to_translate))
                print("model: #%s" % self.model_id)
                print(traceback.format_exc())

                raise ServerModelError(err)

        timer.tick(name="translation")
        self.logger.info(f"""Using model #{self.ct2_model_name} {len(inputs)} inputs
translation time: {timer.times['translation']}""")
        self.reset_unload_timer()

        return predictions, timer.times


    def do_timeout(self):
        """Timeout function that frees GPU memory.

        Moves the model to CPU or unloads it; depending on
        attr`self.on_timemout` value
        """

        if self.on_timeout == "unload":
            self.logger.info("Timeout: unloading model %d" % self.model_id)
            self.unload()
        if self.on_timeout == "to_cpu":
            self.logger.info("Timeout: sending model %d to CPU"
                             % self.model_id)
            self.to_cpu()

    @critical
    def unload(self):
        try:
            del self.translator
            # Set last_used to now() in mt_models table in database
            # print(f"Commited last_used in database for model {self.model_id}")
            db = next(get_db())
            db.query(MT_Model).filter(MT_Model.cbid == str(self.model_id)).update({MT_Model.last_used: datetime.now()})
            db.commit()       
            db.close()
        except Exception as e:
            print("Could not delete translator", e)

        if self.opt['cuda']:
            torch.cuda.empty_cache()
        self.stop_unload_timer()
        self.unload_timer = None
        print(f"Model {self.model_id} fully unloaded")

    def stop_unload_timer(self):
        if self.unload_timer is not None:
            self.unload_timer.cancel()

    def reset_unload_timer(self):
        if self.timeout < 0:
            return

        self.stop_unload_timer()
        self.unload_timer = threading.Timer(self.timeout, self.do_timeout)
        self.unload_timer.start()

    def to_dict(self):
        hide_opt = ["models", "src"]
        d = {"model_id": self.model_id,
             "opt": {k: self.user_opt[k] for k in self.user_opt.keys()
                     if k not in hide_opt},
             "models": self.user_opt["models"],
             "loaded": self.loaded,
             "timeout": self.timeout,
             }
        if self.tokenizers_opt is not None:
            d["tokenizer"] = self.tokenizers_opt
        return d

    @critical
    def to_cpu(self):
        """Move the model to CPU and clear CUDA cache."""
        if type(self.translator) == CTranslate2Translator:
            self.translator.to_cpu()
        else:
            self.translator.model.cpu()
            if self.opt.cuda:
                torch.cuda.empty_cache()

    def to_gpu(self):
        """Move the model to GPU."""
        if type(self.translator) == CTranslate2Translator:
            self.translator.to_gpu()
        else:
            try:
                torch.cuda.set_device('cuda:1')
                print("Tried GPU 1")
                self.translator.model.cuda()
            except:
                try:
                    torch.cuda.set_device('cuda:0')
                    print("Fell back to GPU 0")
                    self.translator.model.cuda()
                except:
                    print("No GPU available")
                    self.translator.model.cpu()

def get_function_by_path(path, args=[], kwargs={}):
    module_name = ".".join(path.split(".")[:-1])
    function_name = path.split(".")[-1]
    try:
        module = importlib.import_module(module_name)
    except ValueError as e:
        print("Cannot import module '%s'" % module_name)
        raise e
    function = getattr(module, function_name)
    return function
