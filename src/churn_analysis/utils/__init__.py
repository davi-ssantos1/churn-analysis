"""general methods to be used during runtime."""

from churn_analysis.utils._models import ModelProtocol
from churn_analysis.utils._sql import execute_query

__all__ = ("ModelProtocol", "execute_query")
