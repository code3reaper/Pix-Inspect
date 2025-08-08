# PixInspect

This is the project that is a solution for the problem statement: **"Image correctness of product on e-marketplaces"**.

## Project Overview

PixInspect matches the product description on the basis of color and type of product by using Google Vision AI and Python modules.

- It extracts the description of products from big e-marketplaces like Amazon and Flipkart using custom-made scrapers.
- The app then matches the product image with the description and rates the image, including quality and resolution.
- To minimize errors from incorrect type or color from the seller side, the seller domain automatically fills in the color and type of product and also checks the image resolution.

## Features
- Automated extraction of product color and type from images using Vision AI
- Scraping product descriptions from Amazon and Flipkart
- Matching and rating product images based on description, color, type, and resolution
- Seller-side automation to reduce manual errors in product listing

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd PixInspect
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Google Vision API Key:**
   - This app uses Google Cloud Vision API. You need a service account key JSON file.
   - **Do NOT commit your API key to the repository.**
   - Set the path to your key as an environment variable:
     ```bash
     set GOOGLE_APPLICATION_CREDENTIALS=path/to/your/vision-api-key.json  # On Windows
     export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/vision-api-key.json  # On Mac/Linux
     ```
   - Replace `path/to/your/vision-api-key.json` with your actual file path.

4. **Run the app:**
   ```bash
   python app.py
   ```

5. **Usage:**
   - Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

**Note:**
- Make sure to replace the Vision API key path with your own credentials.
- Never upload your API key to GitHub or share it publicly. 
