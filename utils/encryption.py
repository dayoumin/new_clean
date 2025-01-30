from cryptography.fernet import Fernet
import os

class FileEncryptor:
    def __init__(self, key=None):
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key

    def encrypt_file(self, input_path, output_path):
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            fernet = Fernet(self.key)
            encrypted_data = fernet.encrypt(data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            return True
        except Exception as e:
            print(f"파일 암호화 실패: {e}")
            return False

    def decrypt_file(self, input_path, output_path):
        try:
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(self.key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            return True
        except Exception as e:
            print(f"파일 복호화 실패: {e}")
            return False

    def save_key(self, key_file):
        with open(key_file, 'wb') as f:
            f.write(self.key)

    def load_key(self, key_file):
        with open(key_file, 'rb') as f:
            self.key = f.read() 