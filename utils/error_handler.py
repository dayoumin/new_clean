import logging
from fastapi import HTTPException
from datetime import datetime
import os

# 로그 디렉토리 설정
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 로그 파일 설정
LOG_FILE = os.path.join(LOG_DIR, f"error_{datetime.now().strftime('%Y%m%d')}.log")

# 로거 설정
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 독립 함수로 log_error 정의
def log_error(error_info: dict):
    """
    에러 정보를 로그 파일에 기록
    :param error_info: 에러 정보를 포함한 딕셔너리
    """
    try:
        timestamp = error_info.get("timestamp", datetime.now().isoformat())
        error = error_info.get("error", {})
        message = error.get("message", "Unknown error")
        stack = error.get("stack", "No stack trace available")
        
        logger.error(f"Timestamp: {timestamp}")
        logger.error(f"Message: {message}")
        logger.error(f"Stack Trace:\n{stack}")
        logger.error("-" * 50)  # 구분선
        
    except Exception as e:
        logger.error(f"Failed to log error: {str(e)}")

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger('error_logger')

    def log_error(self, error_data: dict):
        try:
            timestamp = error_data.get("timestamp", datetime.now().isoformat())
            error = error_data.get("error", {})
            message = error.get("message", "Unknown error")
            stack = error.get("stack", "No stack trace available")
            
            self.logger.error(f"Timestamp: {timestamp}")
            self.logger.error(f"Message: {message}")
            self.logger.error(f"Stack Trace:\n{stack}")
            self.logger.error("-" * 50)  # 구분선
            
        except Exception as e:
            self.logger.error(f"Failed to log error: {str(e)}")

    def handle_http_error(self, status_code: int, detail: str):
        self.log_error({
            'timestamp': datetime.now().isoformat(),
            'error': {
                'message': detail,
                'stack': None
            }
        })
        raise HTTPException(status_code=status_code, detail=detail)

error_handler = ErrorHandler() 