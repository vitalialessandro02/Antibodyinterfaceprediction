import io
import re
import requests
import tarfile
import numpy as np
from config import FIGSHARE_URLS, PHYSICOCHEMICAL_PROPERTIES
from zernike_descriptors import compute_3dzd, map_properties_to_surface, extract_patch
import os
from tqdm import tqdm
from time import sleep
import time

class FigShareLoader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate'
        })
    
    def _get_download_url(self, figshare_url):
        """Obtain direct URL  download from FigShare using API"""
        try:
            
            file_id_match = re.search(r'file=(\d+)', figshare_url)
            if not file_id_match:
                raise ValueError("Cannot extract file ID from the provided URL")
            
            file_id = file_id_match.group(1)
            
            direct_url = f"https://ndownloader.figshare.com/files/{file_id}"
           
            
            # Validate the URL
            try:
                head_response = self.session.head(direct_url, timeout=10)
                head_response.raise_for_status()
           
                return direct_url
            except Exception as e:
                print(f"URL validation warning: {e}")
                
                return direct_url
            
        except Exception as e:
            raise RuntimeError(f"Error obtaining download URL: {str(e)}")

    def _get_tar_stream(self, url, max_retries=3):
     """Obtain direct  stream from file tar.gz """
     for attempt in range(max_retries):
        try:
            download_url = self._get_download_url(url)
           
            response = self.session.get(download_url, stream=True, timeout=600)
            response.raise_for_status()
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            
            content = bytearray()
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                content.extend(data)
            progress_bar.close()
            
            if len(content) != total_size:
                raise RuntimeError(f"Download incomplete: {len(content)}/{total_size} bytes")
            
            # Check for HTML error pages
            if len(content) < 1000 and b'<html' in content[:100].lower():
                raise ValueError("Server returned an HTML error page")
            
            return io.BytesIO(content)
            
        except (requests.exceptions.RequestException, IOError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Download failed after  {max_retries} attemps: {str(e)}")
            
            time.sleep(5 * (attempt + 1)) 

    def _extract_file_from_tar(self, tar_stream, filename_pattern):
        """Extract a specific file from tar.gz store"""
        try:
            tar_stream.seek(0)
            
           
            try:
                with tarfile.open(fileobj=tar_stream, mode='r:gz') as tar:
                    return self._find_and_extract_file(tar, filename_pattern)
            except tarfile.TarError:
               
                tar_stream.seek(0)
                with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                    return self._find_and_extract_file(tar, filename_pattern)
                    
        except tarfile.TarError as e:
            raise RuntimeError(f"Error extracting from tar: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing tar file: {str(e)}")
    
    def _find_and_extract_file(self, tar, filename_pattern):
        """Locate and extract matching file from tar archive"""
        
        file_found = None
        
        for member in tar.getmembers():
            
            if member.isfile() and filename_pattern in member.name:
                file_found = member
                break
        
        if file_found is None:
            raise FileNotFoundError(f"No file correspond  at '{filename_pattern}'")
        
        
        with tar.extractfile(file_found) as file_obj:
            content = file_obj.read()
            
            # Try UTF-8 decoding first
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
            
                try:
                    return content.decode('latin-1')
                except UnicodeDecodeError:
                    
                    return content.decode('utf-8', errors='ignore')

   # remote_data.py
    def _parse_features(self, content, pdb_id=None):
     """Convert the file content  in array numpy"""
  
     data = []
     lines_processed = 0
     lines_skipped = 0
    
     for line_num, line in enumerate(content.split('\n'), 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            # Extract binary labels ( in 0/1)
            label = float(parts[0])
            binary_label = 1 if label == 1 else 0
            
            # Initialize feature vector
            features = np.zeros(276)  
            
            for part in parts[1:]:
                if ':' in part:
                    idx_str, val_str = part.split(':', 1)
                    idx = int(idx_str)
                    val = float(val_str)
                    if 1 <= idx <= 276:
                        features[idx-1] = val
                    else:
                        raise ValueError(f"Feature index out of range: {idx}")
            
            # Add of  data list
            data.append(np.concatenate([[binary_label], features]))
            lines_processed += 1
            
        except Exception as e:
            lines_skipped += 1
            if lines_skipped <= 5:
                print(f"Warning: Skipped line {line_num} - {str(e)}")
            continue
    
     if not data:
        raise ValueError("No valid data found in file")
    
     data_array = np.array(data)
    
     # Verify both classes are present
     unique_labels = np.unique(data_array[:, 0])
     if len(unique_labels) < 2:
        raise ValueError(f"There are only one class in  dataset: {unique_labels}")
    
    # Return only  features e labels 
     return data_array[:, 1:], data_array[:, 0]  # X, y

    def load_features(self, dataset_type, pdb_id=None, return_coords=False):
     """Method to load features from FigShare"""
     if dataset_type not in FIGSHARE_URLS:
        raise ValueError(f"Invalid dataset type. Choose from: {list(FIGSHARE_URLS.keys())}")
    
     
    
     try:
        # 1. Download the tar.gz file
        tar_stream = self._get_tar_stream(FIGSHARE_URLS[dataset_type])
        
        # 2. Extract the descriptors file  
        file_content = self._extract_file_from_tar(tar_stream, "_descriptors_N5.txt")
        
        # 3. Parse the data
        X, y = self._parse_features(file_content)
        
        if return_coords:
            # Generate dummy coordinates if needed
            coords = np.random.rand(len(y), 3) * 100 
            return X, y, coords
        else:
            return X, y
            
     except Exception as e:
        raise RuntimeError(f"Data processing error: {str(e)}")

data_loader = FigShareLoader()