import base64
import os
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Get encryption key from environment variables or use a default key
SECRET_KEY = os.getenv("ENCRYPTION_KEY", "default_encryption_key_for_demonstration_only")

def generate_key(password_provided):
    """
    Generate an encryption key using a password
    
    Args:
        password_provided: Password string
        
    Returns:
        Key for encryption/decryption
    """
    password = password_provided.encode()
    salt = os.getenv("ENCRYPTION_SALT", "default_salt").encode()
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key

def get_encryption_key():
    """
    Get the encryption key from environment or generate one
    
    Returns:
        Encryption key
    """
    key = generate_key(SECRET_KEY)
    return key

def encrypt_data(data):
    """
    Encrypt data using AES encryption
    
    Args:
        data: Data to encrypt (string)
        
    Returns:
        Encrypted data (string)
    """
    if not isinstance(data, str):
        data = str(data)
    
    key = get_encryption_key()
    f = Fernet(key)
    
    # Convert string to bytes
    data_bytes = data.encode()
    
    # Encrypt data
    encrypted_data = f.encrypt(data_bytes)
    
    # Convert encrypted bytes to string for storage
    return base64.urlsafe_b64encode(encrypted_data).decode()

def decrypt_data(encrypted_data):
    """
    Decrypt data encrypted with AES
    
    Args:
        encrypted_data: Encrypted data (string)
        
    Returns:
        Decrypted data (string)
    """
    key = get_encryption_key()
    f = Fernet(key)
    
    # Convert string to bytes
    encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
    
    # Decrypt data
    decrypted_data = f.decrypt(encrypted_bytes)
    
    return decrypted_data.decode()

def anonymize_text(text):
    """
    Anonymize text by removing personal identifiers
    
    Args:
        text: Text to anonymize
        
    Returns:
        Anonymized text
    """
    import re
    
    # Remove common Chinese name patterns
    text = re.sub(r'[\u4e00-\u9fff]{2,3}(?=先生|女士|同志|老师|教授|医生)', '***', text)
    
    # Remove phone numbers (Chinese format)
    text = re.sub(r'1[3-9]\d{9}', '***********', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***@***.***', text)
    
    # Remove ID card numbers (Chinese format)
    text = re.sub(r'\d{17}[\d|X|x]', '******************', text)
    
    # Remove IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '***.***.***.***', text)
    
    return text

def hash_identifier(identifier):
    """
    Create a secure hash of an identifier
    
    Args:
        identifier: Identifier to hash
        
    Returns:
        Hashed identifier
    """
    if not isinstance(identifier, str):
        identifier = str(identifier)
    
    # Create SHA-256 hash
    hash_obj = hashlib.sha256(identifier.encode())
    return hash_obj.hexdigest()

def secure_logging(action, user, data=None):
    """
    Create a secure log entry
    
    Args:
        action: Action performed
        user: User who performed the action
        data: Optional data related to the action
        
    Returns:
        Log entry dictionary
    """
    import time
    
    log_entry = {
        'timestamp': time.time(),
        'action': action,
        'user': user,
        'ip_address': '127.0.0.1'  # In a real system, this would be the actual IP
    }
    
    # If data is provided, encrypt it
    if data:
        log_entry['data_hash'] = hash_identifier(str(data))
    
    return log_entry
