"""AST-guided retrieval and self-correcting patch agent for local Gemma 4 models."""
from .agent import Agent, AgentConfig, AgentResult
from .edits import apply_edits, parse_edits
from .indexer import RepoIndex, Symbol
from .retrieval import make_retriever

__all__ = ["Agent", "AgentConfig", "AgentResult", "RepoIndex", "Symbol",
           "apply_edits", "parse_edits", "make_retriever"]
__version__ = "0.1.0"
