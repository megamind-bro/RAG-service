import os
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import Conversation, Message, ConversationSummary


class ConversationDatabase:
    """Simple JSON-based database for conversation history"""
    
    def __init__(self, db_dir: str = "data/conversations"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_conversation_path(self, conversation_id: str) -> Path:
        """Get the file path for a conversation"""
        return self.db_dir / f"{conversation_id}.json"
    
    def _get_user_index_path(self, user_id: str) -> Path:
        """Get the index file path for a user"""
        return self.db_dir / f"user_{user_id}_index.json"
    
    def create_conversation(
        self, 
        user_id: str, 
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title or "New Conversation",
            messages=[],
            metadata=metadata or {}
        )
        
        # Save conversation
        self._save_conversation(conversation)
        
        # Update user index
        self._update_user_index(user_id, conversation_id)
        
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        path = self._get_conversation_path(conversation_id)
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return Conversation(**data)
    
    def update_conversation(
        self, 
        conversation_id: str, 
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Conversation]:
        """Update conversation metadata"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        if title is not None:
            conversation.title = title
        if metadata is not None:
            conversation.metadata = metadata
        
        conversation.updated_at = datetime.utcnow()
        self._save_conversation(conversation)
        return conversation
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        contexts: Optional[List[dict]] = None
    ) -> Optional[Conversation]:
        """Add a message to a conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        message = Message(
            role=role,
            content=content,
            contexts=contexts
        )
        
        conversation.messages.append(message)
        conversation.updated_at = datetime.utcnow()
        
        # Auto-generate title from first user message if not set
        if conversation.title == "New Conversation" and role == "user" and len(conversation.messages) == 1:
            conversation.title = content[:50] + ("..." if len(content) > 50 else "")
        
        self._save_conversation(conversation)
        return conversation
    
    def get_user_conversations(
        self, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0
    ) -> List[ConversationSummary]:
        """Get all conversations for a user"""
        index_path = self._get_user_index_path(user_id)
        
        if not index_path.exists():
            return []
        
        with open(index_path, 'r', encoding='utf-8') as f:
            conversation_ids = json.load(f)
        
        summaries = []
        for conv_id in conversation_ids:
            conversation = self.get_conversation(conv_id)
            if conversation:
                last_message = None
                if conversation.messages:
                    last_msg = conversation.messages[-1]
                    last_message = last_msg.content[:100] + ("..." if len(last_msg.content) > 100 else "")
                
                summaries.append(ConversationSummary(
                    conversation_id=conversation.conversation_id,
                    user_id=conversation.user_id,
                    title=conversation.title,
                    message_count=len(conversation.messages),
                    created_at=conversation.created_at,
                    updated_at=conversation.updated_at,
                    last_message=last_message
                ))
        
        # Sort by updated_at descending
        summaries.sort(key=lambda x: x.updated_at, reverse=True)
        
        return summaries[offset:offset + limit]
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation or conversation.user_id != user_id:
            return False
        
        # Delete conversation file
        path = self._get_conversation_path(conversation_id)
        if path.exists():
            path.unlink()
        
        # Update user index
        self._remove_from_user_index(user_id, conversation_id)
        
        return True
    
    def _save_conversation(self, conversation: Conversation):
        """Save a conversation to disk"""
        path = self._get_conversation_path(conversation.conversation_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(conversation.model_dump(mode='json'), f, ensure_ascii=False, indent=2, default=str)
    
    def _update_user_index(self, user_id: str, conversation_id: str):
        """Add a conversation to user's index"""
        index_path = self._get_user_index_path(user_id)
        
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                conversation_ids = json.load(f)
        else:
            conversation_ids = []
        
        if conversation_id not in conversation_ids:
            conversation_ids.append(conversation_id)
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_ids, f)
    
    def get_conversation_context(self, conversation_id: str, max_messages: int = 10) -> List[dict]:
        """Get recent conversation context for LLM prompting"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        # Get recent messages for context
        recent_messages = conversation.messages[-max_messages:] if conversation.messages else []
        
        # Convert to simple dict format for LLM
        context = []
        for msg in recent_messages:
            context.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp)
            })
        
        return context
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get a brief summary of the conversation for context"""
        conversation = self.get_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return None
        
        # Create a brief summary from the conversation
        user_messages = [msg.content for msg in conversation.messages if msg.role == "user"]
        if user_messages:
            # Take first few user questions to understand conversation topic
            topics = user_messages[:3]
            return f"Previous topics discussed: {', '.join(topic[:50] for topic in topics)}"
        
        return None
    
    def _remove_from_user_index(self, user_id: str, conversation_id: str):
        """Remove a conversation from user's index"""
        index_path = self._get_user_index_path(user_id)
        
        if not index_path.exists():
            return
        
        with open(index_path, 'r', encoding='utf-8') as f:
            conversation_ids = json.load(f)
        
        if conversation_id in conversation_ids:
            conversation_ids.remove(conversation_id)
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_ids, f)
    
    def get_conversation_context(self, conversation_id: str, max_messages: int = 10) -> List[dict]:
        """Get recent conversation context for LLM prompting"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        # Get recent messages for context
        recent_messages = conversation.messages[-max_messages:] if conversation.messages else []
        
        # Convert to simple dict format for LLM
        context = []
        for msg in recent_messages:
            context.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp)
            })
        
        return context
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get a brief summary of the conversation for context"""
        conversation = self.get_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return None
        
        # Create a brief summary from the conversation
        user_messages = [msg.content for msg in conversation.messages if msg.role == "user"]
        if user_messages:
            # Take first few user questions to understand conversation topic
            topics = user_messages[:3]
            return f"Previous topics discussed: {', '.join(topic[:50] for topic in topics)}"
        
        return None
