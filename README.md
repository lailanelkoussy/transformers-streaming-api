# Transformers-streaming-api
This repository is an example of a streaming response API powered by a huggingface transformers. 
It was built mainly using this [tutorial](https://medium.com/@coldstart_coder/make-responsive-llm-applications-with-text-streaming-56ab045f1425).


This example uses Huggingface **transformers** library and **FastAPI** for the backend, along with a simple **streamlit** frontend to consume the API.


Included is a requirements.txt folder for the env dependencies


# Set up environment
Execute at root of project

````bash
pip install -r requirements.txt
````


# Getting started
This section addresses how to run the code as is.

## Model Backend
At project root, execute :
```bash
uvicorn backend.fastapi_app:app
```

## Streamlit Frontend

At project root, execute :
```bash
streamlit run frontend/streamlit_app.py
```

# Possible Customizations

## Backend
### Change model
Change model name in backend/fastapi_app.py by changing environment variable to other model path

```python
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from StreamingModel import StreamingModel

os.environ['MODEL_NAME'] = 'mistralai/Mistral-7B-v0.3'
```

There are two functions for tokenization in the StreamingModel class, depending on the type of model used. 
If using an Instruct type model with a predefined chat template in the tokenizer (such as mistralai/Mixtral-8x7B-Instruct-v0.1), you can modify the **_start_generation_** function by as such: 

````python

    def start_generation(self, query: str, streamer: CustomStreamer, max_new_tokens=1024, ):
        """
        function *start_generation* starts a generation thread which streams output into *streamer*
        to get model output, get the streamer queue that the streamer was made with, and yield values
        :param query: query to send to model
        :param streamer: streamer to send stream generation into
        :param max_new_tokens : maximum number of new tokens
        :return: None
        """
        + inputs = self.apply_template_and_tokenize(query)
        - # inputs = self.tokenizer([query], return_tensors="pt").to("cuda")
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
````

You can also add more generation arguments from the transformers library for further customization.

## Frontend

The frontend is totally customizable, I simply use Streamlit because it is simple. 
The only crucial part is the client-side consumption of the API in frontend/streamlit_app: 

```python
import json
import requests

def response_generator(prompt, max_new_tokens=1024):
    url = "http://127.0.0.1:8000/query-stream/"
    data = {'query': prompt,
            "max_new_tokens": max_new_tokens}

    with requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"}, stream=True) as r:
        for chunk in r.iter_content(10, decode_unicode=True):
            yield chunk

```

To use in a streamlit setting, it needs to be consumed within an st.write_stream as such: 
```python
import streamlit as st
st.write_stream(response_generator(prompt))
```

Everything else is changeable. 


# Possible error sources 

- If the model seems to be hallucinating/generating rubbish content and you are using several NVIDIA GPUs, you might have a problem with how the model is divided onto your hardware.
  The problem could be originating from the ```device_map='auto'``` parameter in the transformers model declaration.

How to solve this ?
- If you need the model to be partitioned between your GPUs (Does not work on Windows)
  - Make sure you have NCCL (NVIDIA Collective Communications Library) downloaded
  - Try downloading and configuring accelerate using ```bash pip install accelerate``` then ```bash accelerate config```
- If the model can fit on one of your GPUS: 
  - Change transformers declaration as such : 
  ```python 
    - self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, device_map="auto")
  # You can change the 0 in .to(cuda:0) to whichever GPU id you want the model to be loaded to, or you can write .to('cuda')
    + self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to('cuda:0')
  ```


