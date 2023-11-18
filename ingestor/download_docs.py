from typing import List

import requests
from bs4 import BeautifulSoup
import os
import urllib


def main():
    url = "https://www.lemonade.com/faq"
    output_dir = "../lemonade/"
    os.makedirs(output_dir, exist_ok=True)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)
    for link in links:
        href = link

        # If it's a .html file
        if href.endswith(".html"):
            # Make a full URL if necessary
            if not href.startswith("http"):
                href = urllib.parse.urljoin(url, href)

            # Fetch the .html file
            print(f"downloading {href}")
            file_response = requests.get(href)

            file_name = os.path.join(output_dir, os.path.basename(href))
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(file_response.text)


if __name__ == "__main__":
    main()
