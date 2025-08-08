import sys
sys.path.append(r"C:\Users\ASUS\OneDrive\Desktop\SIH RESEMBLER\Final")
import sqlite3
from getpass import getpass

from bs4 import BeautifulSoup
import requests
import time
import os
from fake_useragent import UserAgent
from urllib.parse import urlparse

from google.cloud import vision_v1
from google.cloud.vision_v1 import types

from colormath.color_diff import delta_e_cie1976
from colormath.color_objects import LabColor, sRGBColor
from colormath import color_conversions

from transformers import BertTokenizer, BertModel
import torch
import numpy as np

from color_dict import custom_colors
db_folder = r"C:\Users\ASUS\OneDrive\Desktop\SIH RESEMBLER\Final"
db_file_path = os.path.join(db_folder, 'product_data.db')
conn = sqlite3.connect(db_file_path)

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    url TEXT,
    title TEXT,
    color TEXT,
    image_url TEXT,
    product_type TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
''')

conn.commit()

def check_existing_url(user_id, url):
    cursor.execute('''
    SELECT * FROM products
    WHERE user_id = ? AND url = ?
    ''', (user_id, url))
    result = cursor.fetchone()
    return result

def authenticate_user():
    predefined_user_id = "User"
    
    predefined_password = "123"

    entered_username = input("Enter your username: ")
    entered_password = input("Enter your password: ")

    if (entered_username == predefined_user_id) and (entered_password == predefined_password):
        print("Authentication successful!")
        return predefined_user_id
    else:
        print("Authentication failed. Exiting.")
        conn.close()
        exit()
def insert_product_data(user_id, product_info):
    cursor.execute('''
    INSERT INTO products (user_id, url, title, color, image_url, product_type)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, product_info.get('url'), product_info.get('title'), product_info.get('color'),
          product_info.get('image_url'), product_info.get('product_type')))
    conn.commit()


def get_product_info(url):
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    ua = UserAgent()
    headers = {
        'User-Agent': ua.random,
    }

    try:
        webpage = requests.get(url, headers=headers)
        webpage.raise_for_status()

        soup = BeautifulSoup(webpage.content, 'html.parser')

        
        domain = urlparse(url).hostname
        if 'flipkart' in domain:
            color = get_color_first_flipkart(soup)
            if color is None:
                color = get_color_second_case(soup)
            title = get_title_flipkart(soup)
            image_url = get_image_url_flipkart(soup)
            product_type = get_product_type_flipkart(soup)
        elif 'amazon' in domain:
            color = get_color_amazon(soup)
            title = get_title_amazon(soup)
            image_url = get_image_url_amazon(soup)
            product_type = get_product_type_amazon(soup)
        else:
            print("Unsupported domain.")
            return None

        return {
            'color': color,
            'title': title,
            'image_url': image_url,
            'product_type': product_type
        }

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
#FLIPKART

def get_color_first_flipkart(soup):
    try:
        
        color_info_li_list = soup.find_all("li", class_="_3V2wfe")

       
        image_url = get_image_url_flipkart(soup)

        for color_info_li in color_info_li_list:
            swatch_id = color_info_li.get("id")
            if swatch_id and swatch_id.startswith("swatch-") and swatch_id.endswith("-color"):
                color_name_div = color_info_li.find("div", class_="_2OTVHf _3NVE7n _1mQK5h _2J-DXM")
                color_name = color_name_div.find("div", class_="_3Oikkn _3_ezix _2KarXJ").text.strip()

                color_image_url = color_info_li.find("div", class_="_2C41yO").get("data-img")

                color_image_identifier = color_image_url.split("/")[-1].split(".")[0]

                if color_image_identifier in image_url:
                    print(f"Color: {color_name}")
                    return color_name  

        print("No matching color found. Printing all colors:")
        for color_info_li in color_info_li_list:
            color_name_div = color_info_li.find("div", class_="_2OTVHf _3NVE7n _1mQK5h _2J-DXM")
            color_name = color_name_div.find("div", class_="_3Oikkn _3_ezix _2KarXJ").text.strip()
            print(f"Color: {color_name}")

    except Exception as e:
        print(f"An error occurred while retrieving the color: {e}")

    return None


def get_color_second_case(soup):
    try:
        color_info_li_list = soup.find_all("li", class_="_3V2wfe")

        image_url = get_image_url_flipkart(soup)
        matching_color = None

        for color_info_li in color_info_li_list:
            swatch_id = color_info_li.get("id")
            if swatch_id and swatch_id.startswith("swatch-") and swatch_id.endswith("-color"):
                color_name_div = color_info_li.find("div", class_="_2OTVHf _3NVE7n _1mQK5h _2J-DXM")

                if color_name_div:
                    color_name = color_name_div.find("div", class_="_3Oikkn _3_ezix _2KarXJ _31hAvz")
                    if color_name:
                        color_name = color_name.text.strip()
                        color_image_url = color_info_li.find("div", class_="_2C41yO").get("data-img")

                        color_image_identifier = color_image_url.split("/")[-1].split(".")[0]

                        if color_image_identifier in image_url:
                            matching_color = color_name
                            break

        return matching_color

    except Exception as e:
        print(f"An error occurred while retrieving the color: {e}")

    return None

def get_title_flipkart(soup):
    try:
        title = soup.find("span", class_="B_NuCI").text.strip()
        return title
    except AttributeError:
        return None

def get_image_url_flipkart(soup):
    try:
        div_element = soup.find("div", class_="CXW8mj")

        if div_element:
            img_element = div_element.find("img", class_="_396cs4")

            if img_element:
                img_url = img_element.get('src')

                if img_url:
                    return img_url.strip()
                else:
                    print("Image URL attribute is not present.")
            else:
                print("Image element with class '_396cs4' not found.")
        else:
            print("div element not found.")

            img_element_alt = soup.find("img", class_="_2r_T1I _396QI4")
            if img_element_alt:
                img_url_alt = img_element_alt.get('src')
                if img_url_alt:
                    return img_url_alt.strip()
                else:
                    print("Alternative Image URL attribute is not present.")
            else:
                print("Alternative image element not found.")

    except Exception as e:
        print(f"An error occurred while retrieving the image URL: {e}")

    return None
  

def get_product_type_flipkart(soup):
    try:
        breadcrumb_div = soup.find("div", class_="_1MR4o5")

        breadcrumb_links = breadcrumb_div.find_all("a", class_="_2whKao")

        breadcrumbs = [link.text.strip() for link in breadcrumb_links]

        product_type = breadcrumbs[-4]

        return product_type

    except Exception as e:
        print(f"An error occurred while retrieving the product type: {e}")

    return None
   
#AMAZON
def get_color_amazon(soup):
    color_row = soup.find("tr", class_="a-spacing-small po-color")

    if color_row:
        try:
            color = color_row.find("span", class_="a-size-base po-break-word").text.strip()
        except AttributeError:
            color = ""
    else:
        try:
            color_span = soup.find("span", id="inline-twister-expanded-dimension-text-color_name")
            if color_span:
                color = color_span.text.strip()
            else:
                color = ""
        except AttributeError:
            color = ""

    return color

def get_title_amazon(soup):
    try:
        title = soup.find("span", attrs={'id': 'productTitle'}).text.strip()
        return title
    except AttributeError:
        return None

def get_image_url_amazon(soup):
    try:
        image_element = soup.find("img", attrs={'id': 'landingImage'})

        if image_element:
            image_url = image_element.get('data-old-hires')

            if not image_url:
                image_url = image_element.get('src')

            if image_url:
                return image_url.strip()
            else:
                print("Image URL attribute is not present.")
        else:
            print("Image element not found.")
    except Exception as e:
        print(f"An error occurred while retrieving the image URL: {e}")

    return None

def get_product_type_amazon(soup):
    try:
 
        generic_name_element = soup.find('span', string='Generic Name')

      
        product_type = None
        for sibling in generic_name_element.find_parent('div').find_next_siblings():
            if sibling.name == 'div':
                type_element = sibling.find('span', class_='a-color-base')
                if type_element:
                    product_type = type_element.text.strip()
                break

        if not product_type:
            product_type_element = soup.find('span', class_='a-list-item', text='Generic Name')
            if product_type_element:
                product_type = product_type_element.find_next('span', class_='a-text-bold').find_next('span').text.strip()

        return product_type
    


    except AttributeError:
        return None
save_folder= r"C:\Users\ASUS\OneDrive\Desktop\SIH RESEMBLER\Final\Database\Product Images"
def download_image(url, save_folder):
    try:
        response = requests.get(url)
        response.raise_for_status() 

        parsed_url = urlparse(url)
        filename = os.path.join(save_folder, os.path.basename(parsed_url.path))

        with open(filename, 'wb') as file:
            file.write(response.content)

        print(f"Image downloaded successfully and saved to: {filename}")

        return filename  
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return None
    
def find_closest_color(color_input):
    color_input_lower = color_input.lower()

    if color_input_lower in (name.lower() for name in custom_colors):
        color_name = next(name for name in custom_colors if name.lower() == color_input_lower)
        return custom_colors[color_name]['name'], custom_colors[color_name]['rgb']

    try:
        input_color = sRGBColor(*map(int, color_input.split(',')))

        closest_color = min(custom_colors.items(), key=lambda color: delta_e_cie1976(
            LabColor(*color_conversions.sRGBColor(*color[1]['rgb']).convert_to(color_conversions.LabColor).get_value_tuple()),
            LabColor(*input_color.convert_to(color_conversions.LabColor).get_value_tuple())
        ))

        return closest_color[1]['name'], closest_color[1]['rgb']
    except ValueError:
        return None, None
    
def classify_image(image_path, json_key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_key_path

    client = vision_v1.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    top_labels = []
    for label in labels:
        print(f"{label.description} (confidence: {label.score:.2f})")
        top_labels.append(label.description.lower())  

    response_properties = client.image_properties(image=image)
    colors = response_properties.image_properties_annotation.dominant_colors.colors

    print('\nDominant Colors:')
    if colors:
        top_color_info = colors[0]
        color = top_color_info.color
        print(f"RGB: ({color.red}, {color.green}, {color.blue}), Percentage: {top_color_info.pixel_fraction:.2%}")

    return top_labels


def get_bert_cls_embedding(tokenizer, model, text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    
    cls_embedding = np.array(outputs.last_hidden_state[:, 0, :].squeeze())
    return cls_embedding

def calculate_similarity(embeddings1, embeddings2):
    embeddings1 /= np.linalg.norm(embeddings1)
    embeddings2 /= np.linalg.norm(embeddings2)

    similarity = np.dot(embeddings1, embeddings2)
    return similarity

def scale_similarity_to_percentage(similarity):
    if isinstance(similarity, np.ndarray):
        similarity_scalar = similarity.item() if similarity.size == 1 else similarity
    else:
        similarity_scalar = similarity
    
    percentage_similarity = (similarity_scalar + 1) * 50 
    return percentage_similarity
def print_hex_code(rgb_tuple):
    hex_code = "#{:02x}{:02x}{:02x}".format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])
    print(f"Hex Code: {hex_code}")

def color_difference_percentage(color1, color2):
    if len(color1) != 3 or len(color2) != 3:
        raise ValueError("Input colors must be RGB tuples")

    diff_r = abs(color1[0] - color2[0])
    diff_g = abs(color1[1] - color2[1])
    diff_b = abs(color1[2] - color2[2])

    perc_diff_r = (diff_r / 255.0) * 100
    perc_diff_g = (diff_g / 255.0) * 100
    perc_diff_b = (diff_b / 255.0) * 100

    overall_percentage_difference = (perc_diff_r + perc_diff_g + perc_diff_b) / 3.0

    return overall_percentage_difference   
  
def get_dominant_color(image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\ASUS\OneDrive\Desktop\SIH RESEMBLER\Final\woven-framework-408118-57e3b8267c35.json"

    client = vision_v1.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response_properties = client.image_properties(image=image)
    colors = response_properties.image_properties_annotation.dominant_colors.colors

    if colors:
        top_color_info = colors[0]
        color = top_color_info.color
        return color.red, color.green, color.blue

    return None 

def show_user_history(user_id):
    cursor.execute('''
    SELECT * FROM products
    WHERE user_id = ?
    ''', (user_id,))

    history = cursor.fetchall()

    if not history:
        print("No history found for the user.")
    else:
        print("User History:")
        for entry in history:
            print(f"Product ID: {entry[0]}, URL: {entry[2]}, Title: {entry[3]}, Color: {entry[4]}, Image URL: {entry[5]}, Product Type: {entry[6]}")
      

          
def main():
    user_id = authenticate_user()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    product_url = input("Enter product URL: ")
    existing_product = check_existing_url(user_id, product_url)
    if existing_product:
        print("Product details already exist in the database. Showing details:")
        print(f"Title: {existing_product[3]}")
        print(f"Color: {existing_product[4]}")
        print(f"Image URL: {existing_product[5]}")
        print(f"TYPE: {existing_product[6]}")
        print()
        print("Exiting.")
        conn.close()
        return

    retries = 3  
    for attempt in range(retries):
        product_info = get_product_info(product_url)

        if product_info:
            insert_product_data(user_id, product_info)
            print(f"Title: {product_info.get('title', 'N/A')}")
            print(f"Color: {product_info.get('color', 'N/A')}")
            color_info = product_info.get('color_info', {})
            color_name = color_info.get('name', '').lower()
            closest_color_name, closest_color_rgb = find_closest_color(color_name)
            
            color_name = product_info.get('color', '').lower()
            closest_color_name, closest_color_rgb = find_closest_color(color_name)
            
            if closest_color_name is not None:
                print(f"Closest color to {color_name} is {closest_color_name} with RGB values: {closest_color_rgb}")
            else:
                print(f"RGB values for {color_name} not found or invalid format.")

            print(f"Image URL: {product_info.get('image_url', 'N/A')}")
            print(f"TYPE: {product_info.get('product_type', 'N/A')}")
            print()  

            color_name = product_info.get('color', '').lower()
            closest_color_name, closest_color_rgb = find_closest_color(color_name)

            if closest_color_name is not None:
                print(f"Closest color to {color_name} is {closest_color_name} with RGB values: {closest_color_rgb}")
        
                print_hex_code(closest_color_rgb)
            else:
                print(f"RGB values for {color_name} not found or invalid format.")
            save_folder= r"C:\Users\ASUS\OneDrive\Desktop\SIH RESEMBLER\Final\Database\Product Images"
            image_path = download_image(product_info.get('image_url'), save_folder)
            dominant_color_rgb = get_dominant_color(image_path)

            if dominant_color_rgb is not None:
                print(f"Dominant Color RGB Values: {dominant_color_rgb}")

                color_name = product_info.get('color', '').lower()
                closest_color_name, closest_color_rgb = find_closest_color(color_name)

                if closest_color_rgb is not None:
                    print(f"RGB Values of Closest Color: {closest_color_rgb}")

                    color_diff_percentage = 100-color_difference_percentage(dominant_color_rgb, closest_color_rgb)
                    print(f"Color Difference Percentage: {color_diff_percentage:.2f}%")

                    print("Comparison Result:")
                    if color_diff_percentage < 10.0:
                        print("The dominant color is very similar to the product color or the closest color.")
                    elif color_diff_percentage < 30.0:
                        print("The dominant color is somewhat similar to the product color or the closest color.")
                    else:
                        print("The dominant color is quite different from the product color or the closest color.")
                else:
                    print("Closest color not found or invalid format.")
            else:
                print("Dominant color not found in the image.")
            save_folder= r"C:\Users\ASUS\OneDrive\Desktop\SIH RESEMBLER\Final\Database\Product Images"
            image_path = download_image(product_info.get('image_url'), save_folder)
            json_key_path = r"C:\Users\ASUS\OneDrive\Desktop\SIH RESEMBLER\Final\woven-framework-408118-57e3b8267c35.json"
            top_labels = classify_image(image_path, json_key_path)
            
            embeddings_top_label = get_bert_cls_embedding(tokenizer, model, top_labels[0])
            embeddings_product_type = get_bert_cls_embedding(tokenizer, model, product_info.get('product_type'))

            if embeddings_top_label is not None and embeddings_product_type is not None:
                similarity = calculate_similarity(embeddings_top_label, embeddings_product_type)

                threshold = 0.55  

                percentage_similarity = scale_similarity_to_percentage(similarity)

                if similarity < threshold:
                    print("The items are dissimilar.")
                else:
                    print(f"Similarity Percentage on basis of type: {percentage_similarity:.2f}%")
                    color_weight = 0.7
                    product_type_weight = 0.3

                    total_similarity_percentage = (color_weight * color_diff_percentage) + (product_type_weight * percentage_similarity)

                    print(f"Total Similarity Percentage: {total_similarity_percentage:.2f}%")

                while True:
                    choice = input("Choose an option: (1) Exit, (2) Show History: ")
                    if choice == "1":
                        print("Exiting.")
                        conn.close()
                        return
                    elif choice == "2":
                        show_user_history(user_id)
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
            
            else:
                print("Failed to calculate similarity. One or both embeddings are None.")
            
            break
        else:
            print(f"Failed to retrieve product information. Attempt {attempt + 1}/{retries}")
            time.sleep(5) 

    if not product_info:
        print("Maximum retries reached. Exiting.")

if __name__ == "__main__":
    main()