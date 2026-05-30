"""
LLM-powered document parsing using LangChain and Pydantic.
Extracts structured data from resume and job description text.
"""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import Resume, JobDescription
from utils.utils import get_logger, clean_text, truncate_text

logger = get_logger(__name__)

class ResumeParser:
    """Uses LLM to extract structured data from resume text"""
    
    def __init__(self):
        """Set up LLM, output parser, and prompt template"""
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for LangChain parsing")
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=Resume)
        
        self.prompt_template = PromptTemplate(
            template="""You are an expert resume parser. Given the following resume text, extract structured information.

Resume Text:
{resume_text}

{format_instructions}

Return a valid JSON object that matches the Resume schema exactly.""",
            input_variables=["resume_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def parse_resume(self, text: str, candidate_id: str) -> Dict[str, Any]:
        """Extract structured resume data (name, skills, experience) from text"""
        logger.info(f"parse_resume: parsing candidate_id='{candidate_id}', text_length={len(text)}")

        cleaned_text = clean_text(text)
        cleaned_text = truncate_text(cleaned_text, 8000)
        logger.info(f"parse_resume: cleaned text length={len(cleaned_text)}, invoking LLM ({LLM_MODEL})")

        chain = self.prompt_template | self.llm | self.output_parser

        parsed_resume = chain.invoke({"resume_text": cleaned_text})

        result = parsed_resume.model_dump()
        result.update({
            "candidate_id": candidate_id,
            "raw_text": text,
            "parsing_method": "langchain_structured"
        })

        logger.info(f"parse_resume complete: name='{result.get('name')}', skills={result.get('skills', [])[:5]}, experience={result.get('experience')}")
        return result
    
class JobParser:
    """Uses LLM to extract structured data from job description text"""
    
    def __init__(self):
        """Set up LLM, output parser, and prompt template for job descriptions"""
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for LangChain parsing")
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=JobDescription)
        
        self.prompt_template = PromptTemplate(
            template="""You are an expert job description parser. Given the following job text, extract structured information.

Job Description Text:
{job_text}

{format_instructions}

Return a valid JSON object that matches the JobDescription schema exactly.""",
            input_variables=["job_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def parse_job_description(self, text: str, jd_id: str) -> Dict[str, Any]:
        """Extract structured job data (title, skills, requirements) from text"""
        logger.info(f"parse_job_description: parsing jd_id='{jd_id}', text_length={len(text)}")

        cleaned_text = clean_text(text)
        cleaned_text = truncate_text(cleaned_text, 8000)
        logger.info(f"parse_job_description: cleaned text length={len(cleaned_text)}, invoking LLM ({LLM_MODEL})")

        chain = self.prompt_template | self.llm | self.output_parser

        parsed_jd = chain.invoke({"job_text": cleaned_text})

        result = parsed_jd.model_dump()
        result.update({
            "jd_id": jd_id,
            "raw_text": text,
            "parsing_method": "langchain_structured"
        })

        logger.info(f"parse_job_description complete: title='{result.get('title')}', required_skills={result.get('required_skills', [])[:5]}")
        return result
    