"""
Create a python script to convert pdf file to markdown by using python libraries. (pdf_to_markdown.py)

# check out the format of attachment. Content will be different by pdf files.

# Results should be like: 
1. Clean up spacing and line breaks for readability 
2. Remove page numbers and headers/footers that aren't part of the main content 
3. Keep the content accurate and complete - don't add or remove information 
4. Result file should be stored into the child directory of the command execution(dir name: md_by_local)
"""


---
# llm_pdf_converter.py


#!/usr/bin/env python3
"""
LLM-Powered PDF to Markdown Converter

This application converts PDF files to Markdown format using OpenAI's GPT models
for intelligent text processing and formatting. It extracts text from PDFs and
uses LLM to create well-structured markdown with proper headers, formatting, and organization.
"""

import os
import sys
import fitz  # PyMuPDF
import openai
from pathlib import Path
from typing import List, Optional, Dict
import argparse
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import urllib3
import warnings

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

os.environ["HTTP_PROXY"]="http://70.10.15.10:8080"
os.environ["HTTPS_PROXY"]="http://70.10.15.10:8080"


class LLMPDFToMarkdownConverter:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the converter with OpenAI API credentials.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens_per_request = 4000  # Safe limit for text processing
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text from PDF file page by page.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers as keys and extracted text as values
        """
        try:
            doc = fitz.open(pdf_path)
            pages_content = {}
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Only store non-empty pages
                    pages_content[page_num + 1] = text.strip()
            
            doc.close()
            return pages_content
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, max_length: int = 3000) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        
        Args:
            text: Text to chunk
            max_length: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        # Split by paragraphs first, then by sentences if needed
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
                else:
                    # If single paragraph is too long, split by sentences
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk + sentence) <= max_length:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    if temp_chunk:
                        current_chunk = temp_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def convert_text_to_markdown_with_llm(self, text: str, page_info: str = "") -> str:
        """
        Use OpenAI API to convert raw text to well-formatted markdown.
        
        Args:
            text: Raw text extracted from PDF
            page_info: Additional context about the page
            
        Returns:
            Formatted markdown text
        """
        system_prompt = """You are an expert at converting raw text extracted from PDFs into clean, well-structured Markdown format. 

Your task is to:
1. Identify and properly format headers using ## for main sections and ### for subsections
2. Preserve important formatting like bold, italic, and code blocks where appropriate
3. Convert lists to proper markdown bullet points or numbered lists
4. Format quotes and citations properly
5. Clean up spacing and line breaks for readability
6. Preserve URLs and links
7. Handle tables if present (convert to markdown table format)
8. Remove page numbers and headers/footers that aren't part of the main content
9. Keep the content accurate and complete - don't add or remove information
10. Do not summarize the article. Let the content be as it is.

Return only the formatted markdown content without any explanations or meta-commentary."""

        user_prompt = f"""Convert the following raw text extracted from a PDF into clean, well-structured Markdown format:

{page_info}

Raw text:
{text}

Please format this as clean markdown with appropriate headers, formatting, and structure."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.1  # Low temperature for consistent formatting
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠ LLM processing failed: {str(e)}")
            # Fallback to simple formatting
            return self._simple_markdown_fallback(text)
    
    def _simple_markdown_fallback(self, text: str) -> str:
        """
        Simple fallback formatting if LLM fails.
        
        Args:
            text: Raw text
            
        Returns:
            Basic markdown formatted text
        """
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Basic header detection
            if len(line) < 100 and (line.isupper() or line.istitle()):
                if not any(char.isdigit() for char in line[:10]):
                    formatted_lines.append(f"## {line}\n")
                    continue
            
            formatted_lines.append(f"{line}\n")
        
        return "\n".join(formatted_lines)
    
    def process_pdf_with_llm(self, pdf_path: str, same_directory: bool = True) -> Optional[str]:
        """
        Process a PDF file using LLM for intelligent markdown conversion.
        
        Args:
            pdf_path: Path to PDF file
            same_directory: Whether to save in the same directory as the PDF
            
        Returns:
            Path to generated markdown file if successful, None otherwise
        """
        try:
            print(f"📄 Processing: {pdf_path}")
            
            # Extract text from PDF
            pages_content = self.extract_text_from_pdf(pdf_path)
            
            if not pages_content:
                print(f"⚠ No text content found in {pdf_path}")
                return None
            
            # Generate document header
            pdf_name = Path(pdf_path).stem
            markdown_content = [f"# {pdf_name}\n\n"]
            markdown_content.append(f"*Converted from PDF using LLM on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Process each page
            total_pages = len(pages_content)
            for page_num, page_text in pages_content.items():
                print(f"🤖 Processing page {page_num}/{total_pages} with LLM...")
                
                # Add page separator for multi-page documents
                if page_num > 1:
                    markdown_content.append(f"\n---\n\n")
                
                # Chunk text if it's too long
                text_chunks = self.chunk_text(page_text)
                
                for i, chunk in enumerate(text_chunks):
                    page_info = f"This is content from page {page_num}"
                    if len(text_chunks) > 1:
                        page_info += f", part {i+1} of {len(text_chunks)}"
                    
                    # Convert chunk to markdown using LLM
                    formatted_chunk = self.convert_text_to_markdown_with_llm(chunk, page_info)
                    markdown_content.append(formatted_chunk + "\n\n")
                    
                    # Rate limiting - be nice to the API
                    if len(text_chunks) > 1:
                        time.sleep(1)
            
            # Combine all content
            final_markdown = "".join(markdown_content)
            
            # Generate output path
            markdown_filename = f"{pdf_name}.md"
            if same_directory:
                pdf_directory = Path(pdf_path).parent
                output_path = pdf_directory / markdown_filename
            else:
                output_path = Path.cwd() / markdown_filename
            
            # Save the file
            self.save_markdown_file(final_markdown, str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {str(e)}")
            return None
    
    def save_markdown_file(self, content: str, output_path: str) -> None:
        """
        Save markdown content to file.
        
        Args:
            content: Markdown content
            output_path: Path where to save the file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Markdown saved: {output_path}")
        except Exception as e:
            raise Exception(f"Error saving markdown file: {str(e)}")
    
    def process_multiple_pdfs(self, pdf_paths: List[str], same_directory: bool = True) -> List[str]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            same_directory: Whether to save in the same directory as each PDF
            
        Returns:
            List of successfully processed markdown file paths
        """
        successful_conversions = []
        
        print(f"🚀 Processing {len(pdf_paths)} PDF files with LLM...")
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"\n📋 File {i}/{len(pdf_paths)}")
            
            if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
                result = self.process_pdf_with_llm(pdf_path, same_directory)
                if result:
                    successful_conversions.append(result)
            else:
                print(f"⚠ Skipping invalid file: {pdf_path}")
            
            # Rate limiting between files
            if i < len(pdf_paths):
                print("⏳ Waiting 2 seconds between files...")
                time.sleep(2)
        
        print(f"\n🎉 Successfully processed {len(successful_conversions)} out of {len(pdf_paths)} files")
        return successful_conversions


def main():
    """Main function to run the LLM-powered PDF to Markdown converter."""
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown using OpenAI LLM")
    parser.add_argument("pdf_files", nargs="+", help="PDF files to convert")
    parser.add_argument("--current-dir", action="store_true", 
                       help="Save markdown files in current directory instead of same directory as PDF")
    parser.add_argument("--api-key", 
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="OpenAI model to use (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    load_dotenv()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: OpenAI API key is required!")
        print("Set OPENAI_API_KEY environment variable or use --api-key")
        print("Get your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    # Initialize converter
    try:
        converter = LLMPDFToMarkdownConverter(api_key=api_key, model=args.model)
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {str(e)}")
        sys.exit(1)
    
    # Process files
    same_directory = not args.current_dir
    successful_files = converter.process_multiple_pdfs(
        pdf_paths=args.pdf_files,
        same_directory=same_directory
    )
    
    if successful_files:
        print(f"\n🎉 LLM conversion complete! Generated files:")
        for file_path in successful_files:
            print(f"  ✅ {file_path}")
        
        print(f"\nℹ️  Used model: {args.model}")
        print("💡 The LLM has intelligently formatted your content with proper headers, structure, and formatting!")
    else:
        print("\n❌ No files were successfully converted.")
        sys.exit(1)


if __name__ == "__main__":
    main()



---
# merge03.py

import os
import sys
import argparse
import pdfplumber
from typing import List, Tuple
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def _check_image_quality(img_array: np.ndarray) -> Tuple[bool, str]:
    """이미지 품질 검사
    Args:
        img_array (np.ndarray): 검사할 이미지 배열
    Returns:
        Tuple[bool, str]: 품질 검사 결과 (적합 여부, 메시지)
    """
    h, w = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    edge_density = np.sum(edges > 0) / (h * w)
    if edge_density < 0.01:
        return False, "이미지에 감지 가능한 경계나 내용이 충분하지 않습니다."
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / (h * w)
    non_zero_indices = np.where(hist_normalized > 0.001)[0]
    
    if len(non_zero_indices) > 0:
        contrast_range = non_zero_indices.max() - non_zero_indices.min()
        if contrast_range < 20:
            return False, "이미지 대비가 매우 낮습니다."
    
    return True, "이미지가 경계 상자 검출에 적합합니다."


def _extract_boxes(img_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """이미지에서 경계 상자 좌표 추출"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.Canny(gray, 50, 200)
    
    granularity = 25
    kernel_size = max(1, int(min(img_binary.shape) * granularity / 1000))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(img_binary, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    box_coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_coordinates.append((x, y, x+w, y+h))
        
    return box_coordinates


def _detect_document_type(boxes: List[Tuple[int, int, int, int]]) -> int:
    """문서 유형 감지"""
    if not boxes:
        return 1
    
    x_values = [x for box in boxes for x in (box[0], box[2])]
    y_values = [y for box in boxes for y in (box[1], box[3])]
    
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    page_width = x_max - x_min
    page_height = y_max - y_min
    
    box_centers = [(box[0] + box[2]) / 2 for box in boxes]
    
    num_bins = 20
    hist, bin_edges = np.histogram(box_centers, bins=num_bins, range=(x_min, x_max))
    
    left_region = hist[:num_bins//3].sum()
    middle_region = hist[num_bins//3:2*num_bins//3].sum()
    right_region = hist[2*num_bins//3:].sum()

    is_multi_column = (left_region > middle_region * 0.1 and 
                        right_region > middle_region * 0.1)
    
    if not is_multi_column:
        return 1
    
    header_threshold = y_min + page_height * 0.1
    footer_threshold = y_max - page_height * 0.1
    
    has_header = any(box[1] < header_threshold for box in boxes)
    has_footer = any(box[3] > footer_threshold for box in boxes)
    
    return 3 if has_header or has_footer else 2


def _sort_boxes(boxes: List[Tuple[int, int, int, int]], doc_type: int, tolerance: int = 10) -> List[Tuple[int, int, int, int]]:
    """문서 내 감지된 텍스트 박스들을 문서 유형에 따라 읽기 순서대로 정렬"""
    if not boxes:
        return []
    
    if doc_type == 1:
        return sorted(boxes, key=lambda box: (box[1], box[0]))
    else:
        x_values = [x for box in boxes for x in (box[0], box[2])]
        y_values = [y for box in boxes for y in (box[1], box[3])]
        
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        page_width = x_max - x_min
        page_height = y_max - y_min
        
        header_threshold = y_min + page_height * 0.08
        footer_threshold = y_max - page_height * 0.1
        
        header_boxes = []
        main_boxes = []
        footer_boxes = []
        for box in boxes:
            if box[3] < header_threshold:
                header_boxes.append(box)
            elif box[1] > footer_threshold:
                footer_boxes.append(box)
            else:
                main_boxes.append(box)
        
        column_boundary = x_min + page_width / 2
        left_column = []
        right_column = []
        for box in main_boxes:
            box_center = (box[0] + box[2]) / 2
            if box_center < column_boundary:
                left_column.append(box)
            else:
                right_column.append(box)
        
        sorted_header = sorted(header_boxes, key=lambda box: (box[1], box[0]))
        sorted_left = sorted(left_column, key=lambda box: (box[1], box[0]))
        sorted_right = sorted(right_column, key=lambda box: (box[1], box[0]))
        sorted_footer = sorted(footer_boxes, key=lambda box: box[0])
        
        return sorted_header + sorted_left + sorted_right + sorted_footer


def _process_page(page, use_image_based_extraction: bool = True) -> str:
    """PDF 페이지를 처리하여 텍스트를 추출하는 함수"""
    try:
        x_tolerance = 2
        if not use_image_based_extraction:
            return page.extract_text(x_tolerance=x_tolerance) or ""
        
        image = page.to_image()
        img_array = np.array(image.original)
        
        suitable, _ = _check_image_quality(img_array)
        if not suitable:
            return page.extract_text(x_tolerance=x_tolerance) or ""
        
        boxes = _extract_boxes(img_array)
        doc_type = _detect_document_type(boxes)
        sorted_boxes = _sort_boxes(boxes, doc_type)
        
        texts = []
        for box in sorted_boxes:
            text = page.within_bbox(box).extract_text(x_tolerance=x_tolerance)
            if text:
                texts.append(text)
        
        return "\n".join(texts)
        
    except Exception as e:
        print(f"페이지 처리 중 오류 발생: {str(e)}")
        return page.extract_text(x_tolerance=x_tolerance) or ""


def extract_header_footer_text(path: str, margin: float = 0.12, xtol: float = 10) -> list:
    """PDF 문서에서 머리글 또는 바닥글 영역의 텍스트를 추출하는 함수"""
    string_list = []
    
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            height = page.height
            mheight = height * margin
            
            # 머리글 영역 추출
            crop = page.crop((0, 0, page.width, mheight))
            words = [word['text'] for word in crop.extract_words(x_tolerance=xtol, keep_blank_chars=True)]
            if words:
                string_list = string_list + words
            
            # 바닥글 영역 추출
            crop = page.crop((0, height - mheight, page.width, height))
            words = [word['text'] for word in crop.extract_words(x_tolerance=xtol, keep_blank_chars=True)]
            if words:
                string_list = string_list + words
                
    return string_list


def search_repeated_strings(lines: list[str], threshold_similarity: float = 0.2) -> list:
    """문자열 목록에서 유사도 임계값 기반으로 자주 등장하는 문자열을 찾아내는 함수"""
    if not lines:
        return []
    
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    
    try:
        tfidf_matrix = tfidf.fit_transform(lines)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        repeated_strings = set()
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > threshold_similarity:
                    repeated_strings.add(lines[i])
                    repeated_strings.add(lines[j])
        
        return list(repeated_strings)
    except Exception as e:
        print(f"반복 문자열 검색 중 오류 발생: {str(e)}")
        return []


def remove_newline(text_with_newlines):
    """불필요한 줄바꿈 제거"""
    text_normalized = re.sub(r'\n\n+', '\n', text_with_newlines)
    text_normalized = re.sub(r'\n', ' ', text_normalized)
    return text_normalized


def remove_spaces(text) -> str:
    """연속된 공백 문자를 하나로 통합"""
    text = re.sub(r'[^\S\n]+', ' ', text).strip()
    return text


def normalize_text(text: str, noisewords: list) -> str:
    """텍스트 정규화 수행"""
    # 1. 불필요한 줄바꿈 제거
    text_normalized = remove_newline(text)
    # 2. 불필요한 공백 제거 
    text_normalized = remove_spaces(text_normalized)
    # 3. 불필요 단어 제거
    for word in noisewords:
        if word in text_normalized:
            text_normalized = text_normalized.replace(word, "")
    
    return text_normalized


def extract_title_and_intro(text: str) -> Tuple[str, str]:
    """첫 번째 문장에서 제목을 추출하고 괄호로 시작하는 문장들을 인용문으로 처리"""
    # 전체 텍스트에서 01 Comprehension Questions 전까지의 내용 추출
    title_section_match = re.search(r'^(.*?)(?=01\s*Comprehension Questions|$)', text, re.DOTALL | re.IGNORECASE)
    
    if not title_section_match:
        return "Untitled", ""
    
    title_section = title_section_match.group(1).strip()
    
    # 첫 번째 문장에서 '(' 전까지를 제목으로 추출
    if '(' in title_section:
        title_part = title_section.split('(')[0].strip()
        remaining_part = '(' + '('.join(title_section.split('(')[1:])
    else:
        # '('가 없는 경우 첫 번째 문장을 제목으로 사용
        sentences = re.split(r'[.!?]+', title_section)
        title_part = sentences[0].strip() if sentences else title_section[:100]
        remaining_part = title_section[len(title_part):].strip()
    
    # 남은 부분에서 괄호로 시작하는 문장들을 찾아 인용문으로 변환
    formatted_content = []
    
    if remaining_part:
        # 문장들을 분리
        sentences = re.split(r'(?<=[.!?])\s+', remaining_part)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 괄호로 시작하는 문장을 인용문으로 변환
            if sentence.startswith('('):
                formatted_content.append(f"> {sentence}")
            else:
                formatted_content.append(sentence)
    
    intro_content = '\n\n'.join(formatted_content)
    
    return title_part, intro_content


def identify_sections(text: str) -> List[Tuple[str, str]]:
    """텍스트에서 섹션을 식별하고 분할"""
    # 전체 텍스트에서 섹션 경계 찾기
    full_text = text
    
    # 제목과 소개 부분 추출
    main_title, intro_content = extract_title_and_intro(full_text)
    sections = []
    
    if main_title:
        sections.append(('# ' + main_title, intro_content))
    
    # 나머지 섹션들 찾기
    section_order = [
        (r'01\s*Comprehension Questions', '## 01 Comprehension Questions'),
        (r'02\s*Discussion Questions', '## 02 Discussion Questions'),
        (r'03\s*Vocabulary', '## 03 Vocabulary'),
        (r'04\s*Further Study', '## 04 Further Study: Critical Thinking'),
        (r'05\s*Further Study', '## 05 Further Study: Supplementary content')
    ]
    
    for i, (pattern, header) in enumerate(section_order):
        # 현재 섹션 시작점 찾기
        start_match = re.search(pattern, full_text, re.IGNORECASE)
        if start_match:
            start_pos = start_match.start()
            
            # 다음 섹션 시작점 찾기
            end_pos = len(full_text)
            for next_pattern, _ in section_order[i+1:]:
                next_match = re.search(next_pattern, full_text[start_pos:], re.IGNORECASE)
                if next_match:
                    end_pos = start_pos + next_match.start()
                    break
            
            # 섹션 내용 추출
            section_content = full_text[start_match.end():end_pos].strip()
            if section_content:
                sections.append((header, section_content))
    
    return sections


def format_section_content(content: str, section_type: str) -> str:
    """섹션 유형에 따라 내용을 포맷팅"""
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Vocabulary 섹션의 특별 처리
        if 'Vocabulary' in section_type:
            # 단어: 정의 형태를 **단어**: 정의 형태로 변환
            if ':' in line and not line.startswith('-'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    word = parts[0].strip()
                    definition = parts[1].strip()
                    formatted_lines.append(f"**{word}**: {definition}")
                    continue
        
        # Comprehension Questions, Discussion Questions 섹션의 번호 처리
        if 'Questions' in section_type:
            # 번호가 있는 질문
            if re.match(r'^\d+\.', line):
                formatted_lines.append(f"{line}")
                continue
        
        # 일반적인 리스트 항목
        if re.match(r'^[•·-]', line):
            formatted_lines.append(f"- {line[1:].strip()}")
        # 번호가 있는 리스트
        elif re.match(r'^\d+\.', line):
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n\n'.join(formatted_lines)


def convert_to_markdown(text: str) -> str:
    """텍스트를 섹션별로 분할하여 마크다운 형식으로 변환"""
    sections = identify_sections(text)
    
    if not sections:
        # 섹션을 찾을 수 없는 경우 전체를 제목으로 처리
        first_line = text.split('\n')[0].strip()
        return f"# {first_line}\n\n{text}"
    
    markdown_parts = []
    
    for header, content in sections:
        formatted_content = format_section_content(content, header)
        
        if formatted_content.strip():
            markdown_parts.append(f"{header}\n\n{formatted_content}")
    
    return '\n\n---\n\n'.join(markdown_parts)


def pdf_to_markdown(pdf_path: str, start_page: int = 0, end_page: int = None, 
                   use_image_extraction: bool = True, output_file: str = None) -> str:
    """PDF를 마크다운으로 변환하는 메인 함수"""
    
    print(f"PDF 파일 처리 시작: {pdf_path}")
    
    # 1. 머리글/바닥글에서 노이즈 단어 추출
    print("머리글/바닥글 노이즈 단어 추출 중...")
    header_footer_texts = extract_header_footer_text(pdf_path, margin=0.09, xtol=7)
    noisewords = search_repeated_strings(header_footer_texts, threshold_similarity=0.8)
    
    print(f"발견된 노이즈 단어 수: {len(noisewords)}")
    if noisewords:
        print(f"노이즈 단어 예시: {noisewords[:5]}")
    
    # 2. PDF 페이지별 텍스트 추출 및 정규화
    all_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(end_page, total_pages)
        
        print(f"페이지 {start_page+1}부터 {end_page}까지 처리 중...")
        
        for i, page in enumerate(pdf.pages[start_page:end_page], start_page+1):
            print(f"페이지 {i}/{end_page} 처리 중...", end='\r')
            
            # 텍스트 추출
            text_content = _process_page(page, use_image_extraction)
            
            # 텍스트 정규화
            normalized_text = normalize_text(text_content, noisewords)
            
            if normalized_text.strip():
                all_text.append(normalized_text)
    
    print(f"\n총 {end_page - start_page}페이지 처리 완료")
    
    # 3. 전체 텍스트 결합 후 섹션별 마크다운 변환
    combined_text = ' '.join(all_text)
    markdown_content = convert_to_markdown(combined_text)
    
    # 4. 결과 저장 또는 반환
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"마크다운 파일 저장됨: {output_file}")
    
    return markdown_content


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='PDF를 마크다운으로 변환')
    parser.add_argument('input_file', help='입력 PDF 파일 경로')
    parser.add_argument('-o', '--output', help='출력 마크다운 파일 경로')
    parser.add_argument('-s', '--start', type=int, default=0, help='시작 페이지 (0부터 시작)')
    parser.add_argument('-e', '--end', type=int, help='종료 페이지')
    parser.add_argument('--no-image', action='store_true', help='이미지 기반 추출 비활성화')
    
    args = parser.parse_args()
    
    # 입력 파일 존재 확인
    if not os.path.exists(args.input_file):
        print(f"오류: 파일을 찾을 수 없습니다: {args.input_file}")
        sys.exit(1)
    
    # 출력 파일명 생성 (지정하지 않은 경우)
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = f"{base_name}.md"
    
    try:
        # PDF를 마크다운으로 변환
        markdown_content = pdf_to_markdown(
            pdf_path=args.input_file,
            start_page=args.start,
            end_page=args.end,
            use_image_extraction=not args.no_image,
            output_file=args.output
        )
        
        print(f"변환 완료! 출력 파일: {args.output}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
