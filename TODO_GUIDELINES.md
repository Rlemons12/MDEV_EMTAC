Guidelines for Using Custom TODO Patterns
WHY:
Use for questioning implementation decisions that need clarification:

Unusual code patterns that might confuse new developers
Potential inefficiencies that might have contextual reasons
Legacy approaches that appear outdated
Example: # WHY: Why are we using string concatenation instead of f-strings here?

NO_FUNC:
Use for non-functional code that needs immediate attention:

Critical bugs blocking core functionality
System crashes or fatal errors
Security vulnerabilities requiring immediate patching
Performance issues causing system unavailability
Example: # NO_FUNC: Authentication service fails under load - fix before next commit

PUNCHLIST:
Use for items that must be completed before project completion or release:

Missing features that are in the project scope
Required optimizations
Critical UI/UX adjustments
Documentation that must be completed
Example: # PUNCHLIST: Implement pagination for search results

PENDING:
Use for items waiting on external dependencies:

Features blocked by third-party API integration
Tasks waiting for client/stakeholder decisions
Issues pending review from another team member
Example: # PENDING: Update OAuth flow once client provides new credentials

REVIEW:
Use for code that works but needs examination:

Code that requires performance optimization
Areas that need security review
Complex logic that should be validated by peers
Example: # REVIEW: Check if database query can be optimized

FIXME:
Use for known issues that need repair:

Bugs with identified causes
Performance issues
Security vulnerabilities
Example: # FIXME: Connection pooling causes memory leak under high load

TODO:
Use for general tasks that don't fit other categories:

Minor improvements
Technical debt to address later
Nice-to-have features
Example: # TODO: Add more unit tests for edge cases

Comment Format:
For consistency, always follow this format:
# PATTERN: Brief description [OPTIONAL: Developer initials] [OPTIONAL: Priority (P1-P3)]
Example:
# NO_FUNC: Fix database connection timeout [JS] [P1]
Code Section Marking (Preferred Method):
For marking sections of code that need attention, use PyCharm's region folding:
python# region PATTERN: Description of the issue
def example_function():
    # Code that needs attention
    pass
# endregion
Example:
python# region REVIEW: Authentication methods need security review
def login_user():
    # Login implementation
    pass
# endregion
This creates collapsible code regions that are clearly marked for review.