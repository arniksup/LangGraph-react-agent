from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from dotenv import load_dotenv #из .env-файла подгружаются переменные LangSmith
from langsmith import traceable #простой декоратор для отслеживания
from src.react_agent.graph import graph

import logging
import uvicorn

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Smarty AI Chat Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   #Для продакшена лучше указать конкретные домены
    allow_methods=["*"],
    allow_headers=["*"],
)

@traceable(run_type='llm') #передаём данные в LangSmith
@app.post("/chat")
async def chat_agent(request: Request):
    try:
        request_data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

    session_id = request_data.get("session_id")
    message = request_data.get("message")

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Переменная конфига для выполнения графа
        config = {"configurable": {"thread_id": session_id}}

        # Формируем входящее сообщение
        input_message = {"messages": [{"role": "user", "content": message}]}

        result = await graph.ainvoke(input_message, config=config) # await self.graph.ainvoke(input_message, config=config)

        # Извлекаем последнее сообщение из резалта (должен быть ответ от нейросети)
        if result and "messages" in result and result["messages"]:
            # Берем последнее сообщение
            last_message = result["messages"][-1]

            # Обрабатываем сообщения разного формата
            if hasattr(last_message, 'content'):
                response = last_message.content
            elif isinstance(last_message, tuple) and len(last_message) >= 2:
                response = last_message[1]  # (role, content) tuple
            else:
                response = str(last_message)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

    return {
        "response": response,
        "session_id": session_id
    }

if __name__ == "__main__":
    api_thread = Thread(
        target=lambda: uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        ),
        daemon=True
    )
    api_thread.start()