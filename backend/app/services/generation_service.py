"""
LLM generation service for RAG system
Advanced answer generation with multiple providers
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import json

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from app.models.schemas import RetrievedDocument
from app.config import get_settings
from app.utils.metrics import track_generation_time, increment_counter

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Advanced LLM generation service with multi-provider support
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = None
        self.chat_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM models based on configuration"""
        try:
            logger.info(f"Initializing LLM models: provider={self.settings.LLM_PROVIDER}")
            
            if self.settings.LLM_PROVIDER == "openai":
                self._initialize_openai_models()
            elif self.settings.LLM_PROVIDER == "anthropic":
                self._initialize_anthropic_models()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.settings.LLM_PROVIDER}")
            
            logger.info("LLM models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM models: {e}")
            raise
    
    def _initialize_openai_models(self):
        """Initialize OpenAI models"""
        if not self.settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not provided, using mock responses")
            return
        
        self.chat_model = ChatOpenAI(
            model_name=self.settings.OPENAI_MODEL,
            temperature=self.settings.OPENAI_TEMPERATURE,
            max_tokens=self.settings.OPENAI_MAX_TOKENS,
            openai_api_key=self.settings.OPENAI_API_KEY,
            streaming=True
        )
    
    def _initialize_anthropic_models(self):
        """Initialize Anthropic models"""
        if not self.settings.ANTHROPIC_API_KEY:
            logger.warning("Anthropic API key not provided, using mock responses")
            return
        
        self.chat_model = ChatAnthropic(
            model=self.settings.ANTHROPIC_MODEL,
            anthropic_api_key=self.settings.ANTHROPIC_API_KEY,
            streaming=True
        )
    
    @track_generation_time
    async def generate_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        stream: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive answer using retrieved context
        
        Args:
            query: User query
            retrieved_documents: Retrieved context documents
            stream: Whether to stream the response
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            logger.debug(f"Generating answer for query: {query[:50]}...")
            
            # Prepare context from retrieved documents
            context = await self._prepare_context(retrieved_documents)
            
            # Create system prompt
            system_prompt = await self._create_system_prompt(context)
            
            # Create user prompt
            user_prompt = await self._create_user_prompt(query, conversation_history)
            
            # Generate answer
            if stream:
                answer = await self._generate_streaming_answer(system_prompt, user_prompt)
            else:
                answer = await self._generate_complete_answer(system_prompt, user_prompt)
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                query, answer, retrieved_documents
            )
            
            # Extract source information
            sources_used = [doc.filename for doc in retrieved_documents]
            
            # Prepare result
            result = {
                "answer": answer,
                "confidence_score": confidence_score,
                "sources_used": sources_used,
                "context_length": len(context),
                "generation_metadata": {
                    "model": self.settings.OPENAI_MODEL if self.settings.LLM_PROVIDER == "openai" else self.settings.ANTHROPIC_MODEL,
                    "provider": self.settings.LLM_PROVIDER,
                    "temperature": self.settings.OPENAI_TEMPERATURE,
                    "documents_used": len(retrieved_documents),
                    "prompt_tokens": await self._estimate_tokens(system_prompt + user_prompt)
                }
            }
            
            increment_counter("answers_generated_total", 
                            {"provider": self.settings.LLM_PROVIDER, "streamed": str(stream)})
            
            return result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            increment_counter("generation_errors_total", {"error_type": type(e).__name__})
            raise
    
    async def generate_streaming_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument]
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming answer for real-time response
        """
        try:
            logger.debug(f"Starting streaming generation for: {query[:50]}...")
            
            # Prepare context and prompts
            context = await self._prepare_context(retrieved_documents)
            system_prompt = await self._create_system_prompt(context)
            user_prompt = await self._create_user_prompt(query)
            
            # Stream generation
            if self.chat_model and hasattr(self.chat_model, 'astream'):
                # Real streaming implementation
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                async for chunk in self.chat_model.astream(messages):
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            else:
                # Mock streaming for demo
                answer = await self._generate_mock_answer(query, retrieved_documents)
                
                # Simulate streaming by yielding chunks
                words = answer.split()
                chunk_size = 3
                
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size]) + " "
                    yield chunk
                    await asyncio.sleep(0.1)  # Simulate streaming delay
            
            increment_counter("streaming_answers_generated_total")
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error generating response: {str(e)}"
    
    async def _prepare_context(self, documents: List[RetrievedDocument]) -> str:
        """Prepare context from retrieved documents"""
        try:
            if not documents:
                return "No relevant documents found."
            
            context_parts = []
            total_length = 0
            max_context_length = 4000  # Adjust based on model limits
            
            for i, doc in enumerate(documents):
                # Format document with metadata
                doc_context = f"""[Document {i+1}: {doc.filename}]
Source: {doc.source or 'Unknown'}
Relevance Score: {doc.relevance_score:.3f}

{doc.content}

---"""
                
                # Check if adding this document would exceed limit
                if total_length + len(doc_context) > max_context_length:
                    logger.debug(f"Context length limit reached, using {i} documents")
                    break
                
                context_parts.append(doc_context)
                total_length += len(doc_context)
            
            context = "\n".join(context_parts)
            
            # Add context summary
            summary = f"""CONTEXT SUMMARY:
- Total Documents: {len(context_parts)}
- Total Context Length: {total_length} characters
- Average Relevance Score: {sum(doc.relevance_score for doc in documents[:len(context_parts)]) / len(context_parts):.3f}

RETRIEVED CONTEXT:
{context}"""
            
            return summary
            
        except Exception as e:
            logger.error(f"Context preparation failed: {e}")
            return "Error preparing context from retrieved documents."
    
    async def _create_system_prompt(self, context: str) -> str:
        """Create comprehensive system prompt"""
        system_prompt = f"""You are an advanced AI assistant specializing in answering questions using retrieved context from a knowledge base.

INSTRUCTIONS:
1. **Answer the user's question accurately** using ONLY the information provided in the context below
2. **Cite your sources** by referencing the specific documents (e.g., "According to Document 1..." or "As mentioned in filename.pdf...")
3. **Be comprehensive** but concise - provide thorough answers while staying focused
4. **Maintain accuracy** - if the context doesn't contain enough information, clearly state this limitation
5. **Structure your response** with clear sections and bullet points when appropriate
6. **Provide confidence indicators** - use phrases like "Based on the available information..." when appropriate

RESPONSE GUIDELINES:
- Start with a direct answer to the question
- Support your answer with specific evidence from the context
- Include relevant details and examples from the retrieved documents
- End with a brief summary if the answer is complex
- Always acknowledge the sources of your information

CONTEXT INFORMATION:
{context}

Remember: Base your answer EXCLUSIVELY on the provided context. Do not use external knowledge beyond what's explicitly stated in the retrieved documents."""

        return system_prompt
    
    async def _create_user_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Create user prompt with optional conversation history"""
        user_prompt = f"Question: {query}"
        
        if conversation_history:
            # Add conversation context
            history_text = "\n".join([
                f"Previous Q: {turn['question']}\nPrevious A: {turn['answer'][:200]}..."
                for turn in conversation_history[-3:]  # Last 3 turns
            ])
            user_prompt = f"CONVERSATION HISTORY:\n{history_text}\n\nCURRENT QUESTION: {query}"
        
        return user_prompt
    
    async def _generate_complete_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Generate complete answer using LLM"""
        try:
            if self.chat_model:
                # Real LLM generation
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = await self.chat_model.agenerate([messages])
                answer = response.generations[0][0].text
                
                return answer.strip()
            else:
                # Mock generation for demo
                return await self._generate_mock_answer_from_prompts(system_prompt, user_prompt)
                
        except Exception as e:
            logger.error(f"Complete answer generation failed: {e}")
            raise
    
    async def _generate_streaming_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Generate answer in streaming mode"""
        try:
            # For non-streaming mode, just call complete generation
            return await self._generate_complete_answer(system_prompt, user_prompt)
            
        except Exception as e:
            logger.error(f"Streaming answer generation failed: {e}")
            raise
    
    async def _calculate_confidence_score(
        self,
        query: str,
        answer: str,
        retrieved_documents: List[RetrievedDocument]
    ) -> float:
        """Calculate confidence score for generated answer"""
        try:
            factors = []
            
            # Factor 1: Average retrieval relevance score
            if retrieved_documents:
                avg_relevance = sum(doc.relevance_score for doc in retrieved_documents) / len(retrieved_documents)
                factors.append(avg_relevance)
            
            # Factor 2: Answer length appropriateness
            answer_length_score = min(len(answer) / 500, 1.0)  # Optimal around 500 chars
            factors.append(answer_length_score)
            
            # Factor 3: Source coverage (how many documents were referenced)
            source_coverage = min(len(retrieved_documents) / 3, 1.0)  # Optimal around 3 docs
            factors.append(source_coverage)
            
            # Factor 4: Query-answer alignment (simple keyword overlap)
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            overlap_score = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0
            factors.append(overlap_score)
            
            # Calculate weighted average
            weights = [0.4, 0.2, 0.2, 0.2]  # Prioritize retrieval relevance
            confidence = sum(f * w for f, w in zip(factors, weights))
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence
    
    async def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    async def _generate_mock_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument]
    ) -> str:
        """Generate mock answer for demo purposes"""
        source_info = ", ".join([doc.filename for doc in retrieved_documents[:3]])
        
        answer = f"""Based on my analysis of the retrieved information from {len(retrieved_documents)} relevant documents, here's a comprehensive answer to your question about "{query}":

## Key Findings

{query.title()} is a fundamental topic covered extensively in our knowledge base. The retrieved documents provide detailed insights into this subject matter.

## Detailed Analysis

**Primary Insights:**
- The information from {source_info} indicates that this topic involves multiple interconnected concepts and methodologies
- Current best practices emphasize evidence-based approaches and systematic implementation
- The field continues to evolve with new developments and emerging techniques

**Technical Considerations:**
- Implementation requires careful consideration of various factors including scalability, performance, and maintainability
- Integration with existing systems should follow established patterns and industry standards
- Quality assurance and monitoring are essential for successful deployment

**Practical Applications:**
- Real-world use cases demonstrate the effectiveness of these approaches across different domains
- Success factors include proper planning, adequate resources, and stakeholder alignment
- Common challenges can be mitigated through proper risk assessment and contingency planning

## Sources and References

This answer is based on information retrieved from {len(retrieved_documents)} authoritative sources in the knowledge base, including:
{chr(10).join([f"- {doc.filename} (relevance: {doc.relevance_score:.2f})" for doc in retrieved_documents[:5]])}

The information has been synthesized to provide a comprehensive overview while maintaining accuracy and relevance to your specific question.

*This response demonstrates the RAG system's capability to provide detailed, source-grounded answers using enterprise knowledge.*"""

        return answer
    
    async def _generate_mock_answer_from_prompts(self, system_prompt: str, user_prompt: str) -> str:
        """Generate mock answer from prompts for demo"""
        query = user_prompt.replace("Question: ", "").replace("CURRENT QUESTION: ", "").split("\n")[-1]
        
        return f"""Based on the comprehensive context provided in the knowledge base, here's a detailed response to your inquiry about "{query}":

## Executive Summary

The retrieved documentation provides substantial information addressing your question. Through analysis of multiple authoritative sources, I can provide a well-grounded response that draws from verified information in the knowledge base.

## Detailed Response

**Core Concepts:**
The information indicates that {query.lower()} involves sophisticated methodologies and established best practices. The documentation emphasizes the importance of systematic approaches and evidence-based decision making.

**Implementation Guidelines:**
- Follow established frameworks and proven methodologies
- Ensure proper planning and resource allocation
- Implement comprehensive monitoring and evaluation processes
- Maintain alignment with organizational objectives and standards

**Best Practices:**
- Leverage existing expertise and documented procedures
- Apply appropriate quality assurance measures
- Ensure stakeholder engagement and communication
- Plan for scalability and future growth requirements

**Technical Considerations:**
The technical aspects require careful attention to architecture, performance, security, and integration requirements. The documentation provides specific guidance on implementation patterns and common pitfalls to avoid.

## Conclusion

Based on the retrieved information, successful implementation requires a balanced approach that considers both technical and operational factors. The knowledge base contains comprehensive guidance to support effective decision-making and implementation.

*Note: This response is generated using advanced RAG capabilities, demonstrating the system's ability to provide contextual, source-grounded answers for complex queries.*"""
