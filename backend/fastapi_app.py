import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from StreamingModel import StreamingModel

os.environ['MODEL_NAME'] = 'mistralai/Mistral-7B-v0.3'


class QueryRequest(BaseModel):
    query: str
    max_new_tokens: int


app = FastAPI()
streaming_model = StreamingModel()


@app.post('/query-stream/')
async def stream(request: QueryRequest):
    # Assuming your chat_model.query function can accept max_new_tokens as an argument
    return StreamingResponse(streaming_model.query(request.query, max_new_tokens=request.max_new_tokens),
                             media_type='text/event-stream')
