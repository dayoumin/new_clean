import time
from prometheus_client import start_http_server, Counter, Histogram
from models.chat_models import ChatRequest

class Monitoring:
    def __init__(self):
        self.REQUEST_COUNT = Counter(
            'chat_requests_total',
            'Total number of chat requests',
            ['endpoint', 'status']
        )
        
        self.REQUEST_LATENCY = Histogram(
            'chat_request_latency_seconds',
            'Latency of chat requests',
            ['endpoint']
        )
        
        self.ERROR_COUNT = Counter(
            'chat_errors_total',
            'Total number of chat errors',
            ['error_type']
        )

    def start_metrics_server(self, port=8000):
        start_http_server(port)

    def track_request(self, endpoint):
        def decorator(func):
            async def wrapper(chat_request: ChatRequest):
                start_time = time.time()
                try:
                    result = await func(chat_request)
                    self.REQUEST_COUNT.labels(endpoint=endpoint, status='success').inc()
                    return result
                except Exception as e:
                    self.REQUEST_COUNT.labels(endpoint=endpoint, status='error').inc()
                    self.ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                    raise
                finally:
                    latency = time.time() - start_time
                    self.REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
            return wrapper
        return decorator

monitoring = Monitoring() 