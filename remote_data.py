# remote_data.py
import io
import re
import requests
import tarfile
import numpy as np
from config import FIGSHARE_URLS

class FigShareLoader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate'
        })
    
    def _get_download_url(self, figshare_url):
        """Ottiene l'URL diretto di download da FigShare usando l'API"""
        try:
            print(f"Ottenendo URL di download da: {figshare_url}")
            
            # Estrai l'ID del file dall'URL
            file_id_match = re.search(r'file=(\d+)', figshare_url)
            if not file_id_match:
                raise ValueError("Impossibile estrarre l'ID del file dall'URL")
            
            file_id = file_id_match.group(1)
            print(f"ID del file: {file_id}")
            
            # Costruisci l'URL diretto usando il servizio ndownloader di FigShare
            direct_url = f"https://ndownloader.figshare.com/files/{file_id}"
            print(f"URL di download diretto: {direct_url}")
            
            # Verifica che l'URL sia valido facendo una richiesta HEAD
            try:
                head_response = self.session.head(direct_url, timeout=10)
                head_response.raise_for_status()
                print(f"URL verificato - Content-Type: {head_response.headers.get('Content-Type', 'Non specificato')}")
                print(f"Content-Length: {head_response.headers.get('Content-Length', 'Non specificato')}")
                return direct_url
            except Exception as e:
                print(f"Errore nella verifica dell'URL: {e}")
                # Se la verifica fallisce, prova comunque l'URL
                return direct_url
            
        except Exception as e:
            raise RuntimeError(f"Errore nell'ottenere l'URL di download: {str(e)}")

    def _get_tar_stream(self, url):
        """Ottiene uno stream diretto al file tar.gz"""
        try:
            # Ottieni l'URL diretto di download
            download_url = self._get_download_url(url)
            
            print(f"Downloading da: {download_url}")
            response = self.session.get(download_url, stream=True, timeout=120)
            response.raise_for_status()
            
            print(f"Content-Type: {response.headers.get('Content-Type', 'Non specificato')}")
            print(f"Content-Length: {response.headers.get('Content-Length', 'Non specificato')}")
            
            # Leggi tutto il contenuto
            content = response.content
            print(f"Dimensione contenuto scaricato: {len(content)} bytes")
            
            # Verifica che non sia una pagina di errore HTML
            if len(content) < 1000 and b'<html' in content[:100].lower():
                raise ValueError("Il contenuto sembra essere una pagina HTML di errore")
            
            # Crea uno stream in memoria
            tar_stream = io.BytesIO(content)
            
            # Verifica che sia un file tar.gz provando ad aprirlo
            tar_stream.seek(0)
            try:
                with tarfile.open(fileobj=tar_stream, mode='r:gz') as tar:
                    # Se arriviamo qui, è un file tar.gz valido
                    print("File tar.gz verificato con successo")
                    tar_stream.seek(0)  # Riposiziona all'inizio
                    return tar_stream
            except tarfile.TarError as e:
                # Se non è un tar.gz, prova senza compressione
                tar_stream.seek(0)
                try:
                    with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                        print("File tar (non compresso) verificato con successo")
                        tar_stream.seek(0)
                        return tar_stream
                except tarfile.TarError:
                    # Come ultimo tentativo, salva i primi 200 bytes per debug
                    tar_stream.seek(0)
                    debug_content = tar_stream.read(200)
                    print(f"Debug - Primi 200 bytes: {debug_content}")
                    raise ValueError(f"Il contenuto non è un file tar o tar.gz valido: {str(e)}")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Errore nel download: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Errore nel processamento del download: {str(e)}")

    def _extract_file_from_tar(self, tar_stream, filename_pattern):
        """Estrae un file specifico dall'archivio tar.gz"""
        try:
            tar_stream.seek(0)
            
            # Prova prima con compressione gzip
            try:
                with tarfile.open(fileobj=tar_stream, mode='r:gz') as tar:
                    return self._find_and_extract_file(tar, filename_pattern)
            except tarfile.TarError:
                # Se fallisce, prova senza compressione
                tar_stream.seek(0)
                with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                    return self._find_and_extract_file(tar, filename_pattern)
                    
        except tarfile.TarError as e:
            raise RuntimeError(f"Errore nell'estrazione del tar: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Errore generico nell'estrazione: {str(e)}")
    
    def _find_and_extract_file(self, tar, filename_pattern):
        """Trova ed estrae un file dal tar"""
        print("File nell'archivio:")
        file_found = None
        
        for member in tar.getmembers():
            print(f"  - {member.name}")
            if member.isfile() and filename_pattern in member.name:
                file_found = member
                break
        
        if file_found is None:
            raise FileNotFoundError(f"Nessun file corrisponde a '{filename_pattern}'")
        
        print(f"Estrazione del file: {file_found.name}")
        with tar.extractfile(file_found) as file_obj:
            content = file_obj.read()
            
            # Prova a decodificare come UTF-8
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                # Se fallisce, prova con latin-1
                try:
                    return content.decode('latin-1')
                except UnicodeDecodeError:
                    # Come ultimo tentativo, ignora i caratteri problematici
                    return content.decode('utf-8', errors='ignore')

    def _parse_features(self, content):
        """Converte il contenuto del file in array numpy"""
        print("Parsing dei dati...")
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
                    
                label = float(parts[0])
                # Accetta sia -1 che 1 come etichette valide
                if label not in [-1, 1]:
                    raise ValueError(f"Etichetta non valida: {label}. Sono accettate solo -1 e 1.")
                
                # Converti le etichette da [-1, 1] a [0, 1] per la classificazione binaria
                binary_label = 1 if label == 1 else 0
                
                features = np.zeros(276)  # 276 features come nel paper
                
                for part in parts[1:]:
                    if ':' in part:
                        idx_str, val_str = part.split(':', 1)
                        idx = int(idx_str)
                        val = float(val_str)
                        if 1 <= idx <= 276:  # Verifica che l'indice sia valido
                            features[idx-1] = val
                        else:
                            raise ValueError(f"Indice feature fuori range: {idx}")
                
                data.append(np.concatenate([[binary_label], features]))
                lines_processed += 1
                
            except Exception as e:
                lines_skipped += 1
                if lines_skipped <= 5:  # Mostra solo i primi 5 errori
                    print(f"Warning: Saltata linea {line_num} - {str(e)}")
                continue
        
        print(f"Linee processate: {lines_processed}")
        print(f"Linee saltate: {lines_skipped}")
        
        if not data:
            raise ValueError("Nessun dato valido trovato nel file")
        
        data_array = np.array(data)
        print(f"Dimensione dati: {data_array.shape}")
        
        # Verifica che ci siano entrambe le classi
        unique_labels = np.unique(data_array[:, 0])
        if len(unique_labels) < 2:
            raise ValueError(f"Solo una classe presente nel dataset: {unique_labels}")
        
        return data_array[:, 1:], data_array[:, 0]  # X, y

    def load_features(self, dataset_type):
        """Carica i dati direttamente da FigShare"""
        if dataset_type not in FIGSHARE_URLS:
            raise ValueError(f"Tipo dataset non valido. Scegli tra: {list(FIGSHARE_URLS.keys())}")
        
        print(f"Accesso al dataset {dataset_type} da FigShare...")
        
        try:
            # 1. Ottieni lo stream del file tar.gz
            tar_stream = self._get_tar_stream(FIGSHARE_URLS[dataset_type])
            
            # 2. Estrai il file specifico
            file_content = self._extract_file_from_tar(tar_stream, "_descriptors_N5.txt")
            
            # 3. Processa i dati
            return self._parse_features(file_content)
            
        except Exception as e:
            raise RuntimeError(f"Errore nel processamento: {str(e)}")

# Istanza globale
data_loader = FigShareLoader()