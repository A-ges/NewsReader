!pip install -q -U google-generativeai
import pathlib
import textwrap
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from google.colab import userdata

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = genai.GenerativeModel('gemini-2.5-flash')
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

article_url = 'https://ground.news/article/vance-heads-to-israel-to-shore-up-gaza-truce-deal-in-talks-with-netanyahu'   #Input the article link here

try:
    response = requests.get(article_url)
    response.raise_for_status() # Raise an exception for bad status codes

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the only relevant text content (no ads)
    article_text = ""
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        article_text += p.get_text() + "\n"

    # Generate article summary
    summary_prompt = ["Write a summary of the content of this news article.", article_text]     #Have to change this prompt according to frontend user input
    summary_response = model.generate_content(summary_prompt, stream=True)
    summary_response.resolve()
    print("Article Summary:")
    display(to_markdown(summary_response.text))
    print("\n---\n") # Separator

    # Extract and describe images
    images = soup.find_all('img')
    print("Image Descriptions:")
    for img in images:
        img_url = img.get('src')
        if img_url:
            # Construct absolute URL if necessary
            if not img_url.startswith('http'):
                img_url = article_url.rsplit('/', 1)[0] + '/' + img_url

            alt_text = img.get('alt', 'An image from the article.')

            # Find the closest paragraph before the image for relevant text
            relevant_text = ""
            previous_sibling = img.find_previous_sibling()
            while previous_sibling and previous_sibling.name != 'p':
                previous_sibling = previous_sibling.find_previous_sibling()
            if previous_sibling and previous_sibling.name == 'p':
                relevant_text = previous_sibling.get_text()

            # Use the generative model to describe the image in context
            image_prompt = f"Briefly describe this image from a news article for someone who cannot see it.
            Relate the description to the surrounding text: {relevant_text}.
            Image Description (if available): {alt_text}
            Image URL: {img_url}
            Description:
            """
            image_response = model.generate_content(image_prompt)
            print(f"Image: {alt_text}")
            display(to_markdown(image_response.text))
            print("\n")

except requests.exceptions.RequestException as e:
    print(f"Error fetching article from URL: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#Example Usage:
#response = model.generate_content("What is Software Engineering for CSAI?")
#to_markdown(response.text)
