#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:43:41 2023

@author: liyingqiu

This file is used to extract sentences from pdf files.
"""

import os
from PyPDF2 import PdfReader

def extract_sentence(pdf_path):

    reader = PdfReader(pdf_path)

    num_pages = len(reader.pages)

    content = ""

    for page_number in range(num_pages):

        page_content = reader.pages[page_number].extract_text()

        # print(page_content)

        content += page_content
    
    return content






