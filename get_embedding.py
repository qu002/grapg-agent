import json
import httpx
import numpy as np
from typing import Literal

from tenacity import retry, stop_after_attempt, wait_exponential
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def get_embeddings_from_httpx(
    data: list, 
    endpoint: Literal["embed_text", "embed_image"]  # 限制端点类型
):
    
    async with httpx.AsyncClient() as client:
        try:
            if "text" in endpoint:
                response = await client.post(
                    f"http://localhost:8005/{endpoint}",
                    json={"queries": data}, 
                    timeout=120.0  # 根据文件大小调整超时
                )
            else:
                response = await client.post(                      #图像的嵌入，从本地服务端的地址8005端口发出post请求9（发送数据--图像）
                    f"http://localhost:8005/{endpoint}",
                    #json={payload_key: data}  # 动态字段名
                    files=data,                                   #发送的数据要符合本地服务器要求的数据格式？？什么格式呢？--列表
                    timeout=120.0  # 根据文件大小调整超时
                )
            response.raise_for_status()                           #检查响应状态码
            return np.array(response.json()["embeddings"])       #解析响应json，提取embedding   嵌入向量转换为numpy数组返回
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"HTTP request failed: {e}")