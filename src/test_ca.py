import os
import ssl

import certifi

print(certifi.where())
print(ssl.get_default_verify_paths().openssl_cafile)

openssl_dir, openssl_cafile = os.path.split(
    ssl.get_default_verify_paths().openssl_cafile
)
# no content in this folder
os.listdir(openssl_dir)
# non existent file
print(os.path.exists(os.path.join(openssl_dir, openssl_cafile)))
