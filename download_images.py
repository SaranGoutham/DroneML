import os
import requests
from duckduckgo_search import DDGS

def download_images(query, max_images=20, save_dir='data'):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Searching for: {query}")
    with DDGS() as ddgs:
        results = list(ddgs.images(
            query,
            max_results=max_images
        ))
        
    if not results:
        print("No images found.")
        return
        
    print(f"Found {len(results)} images. Downloading...")
    
    downloaded = 0
    for i, res in enumerate(results):
        image_url = res.get('image')
        if not image_url:
            continue
            
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Determine extension
            ext = '.jpg'
            if 'png' in image_url.lower():
                ext = '.png'
                
            filepath = os.path.join(save_dir, f"thermal_landmine_{i+1}{ext}")
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            print(f"Downloaded: {filepath}")
            downloaded += 1
            
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")
            
    print(f"\nSuccessfully downloaded {downloaded} images to {save_dir}")

if __name__ == "__main__":
    download_images("thermal images of buried land mines", max_images=30, save_dir=r"d:\DroneML\data\thermal_landmines")
