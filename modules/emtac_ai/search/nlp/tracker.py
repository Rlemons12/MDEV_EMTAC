class SearchQueryTracker:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def start_session(self, user_id: str, context_data: Optional[Dict] = None) -> int:
        ...

    def record_query(self, session_id: int, query_text: str, intent: str, entities: List[str]):
        ...

    def record_click(self, session_id: int, query_id: int, result_id: str):
        ...
