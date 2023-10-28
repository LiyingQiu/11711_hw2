#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:12 2023

@author: liyingqiu
"""

import requests
from bs4 import BeautifulSoup
import os

def download_pdf(url, save_path):

    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"PDF downloaded successfully to {save_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")

def extract_paper_link(url):

    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    # find the class which is "d-sm-flex align-items-stretch"
    paper_list = soup.find_all("p", class_="d-sm-flex align-items-stretch")

    # find the sub element in paper_list which has href
    paper_link = [paper.find("a") for paper in paper_list]

    # extract the link from href
    paper_link = [link["href"] for link in paper_link]

    return paper_link


def get_all_pdf(conf = "acl", year = "2023"):

    corpus_url = "https://aclanthology.org/events/" + conf + "-" + year + "/"

    paper_links = extract_paper_link(corpus_url)

    print(f"The number of papers in {conf}-{year} is: {len(paper_links)}.")

    for i in range(1, len(paper_links)):

        paper_link = paper_links[i]

        file_name = paper_link.split("org/")[1]

        if not os.path.exists("./Data/" + file_name):

            download_pdf(paper_link,  "./Data/"+ file_name)


get_all_pdf(conf = "acl", year = "2023")
get_all_pdf(conf = "acl", year = "2022")
get_all_pdf(conf = "acl", year = "2021")
get_all_pdf(conf = "emnlp", year = "2022")
get_all_pdf(conf = "emnlp", year = "2021")
get_all_pdf(conf = "naacl", year = "2022")
get_all_pdf(conf = "naacl", year = "2021")



