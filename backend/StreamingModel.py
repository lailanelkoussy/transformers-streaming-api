from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
from CustomStreamer import CustomStreamer
import os
from queue import Queue
import asyncio


class StreamingModel:

    def __init__(self):

        self.model_name = os.environ.get("MODEL_NAME")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, skip_special_tokens=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.streamer_queue = Queue()
        self.streamer = CustomStreamer(self.streamer_queue, self.tokenizer, True)

    def apply_template_and_tokenize(self, query: str) -> torch.Tensor:
        """

        :param query: string query
        :return: torch tensor of input query put in chat template format
        """
        messages = [{"role": "user", "content": query}]
        chat_template = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer([chat_template], return_tensors="pt").to("cuda")
        return inputs

    def start_generation(self, query: str, streamer: CustomStreamer, max_new_tokens=1024, ):
        """
        function *start_generation* starts a generation thread which streams output into *streamer*
        to get model output, get the streamer queue that the streamer was made with, and yield values
        :param query: query to send to model
        :param streamer: streamer to send stream generation into
        :param max_new_tokens : maximum number of new tokens
        :return: None
        """
        # inputs = self.apply_template_and_tokenize(query)
        inputs = self.tokenizer([query], return_tensors="pt").to("cuda")
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

    async def query(self, query: str, max_new_tokens=1024):
        """
        function *query* is asynchronous, it yields real-time generation response from model
        :param query: string query
        :return: async string response
        """
        self.start_generation(query, self.streamer, max_new_tokens)

        # Infinite loop
        while True:
            # Retrieving the value from the queue
            value = self.streamer_queue.get()
            # Breaks if a stop signal is encountered
            if value is None:
                break
            # yields the value
            value = value.replace('</s>', '')
            yield value

            # provides a task_done signal once value yielded
            self.streamer_queue.task_done()
            # guard to make sure we are not extracting anything from
            # empty queue
            await asyncio.sleep(0.1)
