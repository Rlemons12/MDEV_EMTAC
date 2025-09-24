# Guidelines for Using Custom TODO Patterns

Use these standardized tags in code comments to improve clarity, prioritization, and tracking. All tags are highlighted with unique colors/effects in PyCharm for quick identification.

---

## PATTERN OVERVIEW

| Pattern       | Example                                                                   | Use Case                                                        | Recommended Color/Effect             |
|---------------| ------------------------------------------------------------------------- |-----------------------------------------------------------------| ------------------------------------ |
| WHY           | `# WHY: Why are we using string concatenation instead of f-strings here?` | Questionable or non-obvious implementation; needs clarification | **Teal**, *italic*, underline        |
| NO\_FUNC      | `# NO_FUNC: Authentication service fails under load - fix ASAP`           | Critical, non-functional code; must-fix bugs or blockers        | **Red**, **bold**, boxed             |
| PUNCHLIST     | `# PUNCHLIST: Implement pagination for search results`                    | Must-complete items before release or project end               | **Orange**, bold, wavy underline     |
| PENDING       | `# PENDING: Update OAuth flow once client provides credentials`           | Items waiting on external dependencies                          | **Grey**, italic, dotted underline   |
| REVIEW        | `# REVIEW: Check if DB query can be optimized`                            | Works, but needs peer or expert review                          | **Blue**, bold, underline            |
| REVIEW/REMOVE | `# REVIEW/REMOVE: Legacy validation - check if still needed before delete` | Code that needs review to determine if it can be safely removed | **Purple**, bold, dashed underline   |
| ENHANCEMENT   | `# ENHANCEMENT: Add refresh button for real-time updates`                 | Consider for upgrade, future improvement, or refactor           | **Green**, bold, wavy underline      |
| FIXME         | `# FIXME: Connection pooling causes memory leak under load`               | Known issues that need to be fixed                              | **Orange-Red**, bold, wavy underline |
| TODO          | `# TODO: Add more unit tests for edge cases`                              | General tasks, minor improvements, or technical debt            | **Sky Blue**, none or underline      |
| REMOVE        | `# REMOVE?: This block looks redundant — verify before deleting`          | Suspected unnecessary code; candidate for removal               | **Purple**, bold, dashed underline   |  
 
---

### Detailed Tag Guidelines & Color Schemes

#### WHY

* **When to Use:**
  For questioning implementation decisions that need clarification:

  * Unusual code patterns that might confuse new developers
  * Potential inefficiencies that might have contextual reasons
  * Legacy approaches that appear outdated
* **Example:**
  `# WHY: Why are we using string concatenation instead of f-strings here?`
* **Color/Effect:**

  * **Teal** (`#008080`)
  * *Italic*
  * **Underline**

---

#### NO\_FUNC

* **When to Use:**
  For non-functional code that needs immediate attention:

  * Critical bugs blocking core functionality
  * System crashes or fatal errors
  * Security vulnerabilities requiring immediate patching
  * Performance issues causing system unavailability
* **Example:**
  `# NO_FUNC: Authentication service fails under load - fix before next commit`
* **Color/Effect:**

  * **Red** (`#D32F2F`)
  * **Bold**
  * **Boxed**

---

#### PUNCHLIST

* **When to Use:**
  For items that must be completed before project completion or release:

  * Missing features in project scope
  * Required optimizations
  * Critical UI/UX adjustments
  * Documentation that must be completed
* **Example:**
  `# PUNCHLIST: Implement pagination for search results`
* **Color/Effect:**

  * **Orange** (`#FF9800`)
  * **Bold**
  * **Wavy underline**

---

#### PENDING

* **When to Use:**
  For items waiting on external dependencies:

  * Features blocked by third-party API integration
  * Tasks waiting for client/stakeholder decisions
  * Issues pending review from another team member
* **Example:**
  `# PENDING: Update OAuth flow once client provides new credentials`
* **Color/Effect:**

  * **Grey** (`#757575`)
  * *Italic*
  * **Dotted underline**

---

#### REVIEW

* **When to Use:**
  For code that works but needs examination:

  * Performance optimization
  * Security review
  * Complex logic validation by peers
* **Example:**
  `# REVIEW: Check if database query can be optimized`
* **Color/Effect:**

  * **Blue** (`#1976D2`)
  * **Bold**
  * **Underline**

---

#### REVIEW/REMOVE

* **When to Use:**
  For code that needs review to determine if it can be safely removed:

  * Legacy code that may no longer be needed
  * Dead code paths that appear unused
  * Deprecated functions that might still have dependencies
  * Redundant implementations that need verification before deletion
  * Code marked for removal that requires peer confirmation
* **Example:**
  `# REVIEW/REMOVE: Legacy validation logic - verify no dependencies before removing`
* **Color/Effect:**

  * **Purple** (`#9C27B0`)
  * **Bold**
  * **Dashed underline**

---

#### ENHANCEMENT

* **When to Use:**
  For future improvements, enhancements, or refactor opportunities:

  * Nice-to-have features for next upgrade/release
  * Non-urgent code improvements
  * Optional UI/UX additions
* **Example:**
  `# ENHANCEMENT: Add refresh button for better UX`
* **Color/Effect:**

  * **Green** (`#388E3C`)
  * **Bold**
  * **Wavy underline**

---

#### FIXME

* **When to Use:**
  For known issues that need repair:

  * Bugs with identified causes
  * Performance issues
  * Security vulnerabilities
* **Example:**
  `# FIXME: Connection pooling causes memory leak under high load`
* **Color/Effect:**

  * **Orange-Red** (`#FF5722`)
  * **Bold**
  * **Wavy underline**

---

#### TODO

* **When to Use:**
  For general tasks that don't fit other categories:

  * Minor improvements
  * Technical debt
  * Nice-to-have features
* **Example:**
  `# TODO: Add more unit tests for edge cases`
* **Color/Effect:**

  * **Sky Blue** (`#0288D1`)
  * **Underline** (optional)

---

#### REMOVE

* **When to Use:**
  For code that is suspected to be unnecessary but needs verification:

  * Code that appears redundant
  * Functions that seem unused
  * Experimental code that may be safe to delete
* **Example:**
  `# REMOVE?: This validation block looks redundant — verify before deleting`
* **Color/Effect:**

  * **Purple** (`#9C27B0`)
  * **Bold**
  * **Dashed underline**

---

## Comment Format

For consistency, always use:

```
# PATTERN: Brief description [OPTIONAL: Developer initials] [OPTIONAL: Priority (P1-P3)]
```

**Example:**

```python
# REVIEW/REMOVE: Legacy user validation method [JS] [P2]
```

---

## Code Section Marking (Preferred Method)

For larger sections needing attention, use PyCharm's region folding:

```python
# region REVIEW/REMOVE: Legacy authentication methods - verify dependencies before removal
def old_login_method():
    # Code that might be removable
    pass
# endregion
```

This creates a collapsible code region marked for review and potential removal.

---

## How to Set These in PyCharm

1. **Settings** → `Editor` → `TODO` → Add patterns.
2. For each, set the:

   * **Pattern** (e.g., `REVIEW/REMOVE[:]`)
   * **Color** (choose hex code above)
   * **Effects** (bold, underline, wavy underline, boxed, etc.)

---

### Summary Table (Copy for Team Wiki)

| Pattern       | Color      | Style/Effect         | Example                                               |
| ------------- | ---------- | -------------------- | ----------------------------------------------------- |
| WHY           | Teal       | Italic, Underline    | `# WHY: Why do we do it this way?`                    |
| NO\_FUNC      | Red        | Bold, Boxed          | `# NO_FUNC: Fix authentication failure ASAP`          |
| PUNCHLIST     | Orange     | Bold, Wavy Underline | `# PUNCHLIST: Implement pagination before release`    |
| PENDING       | Grey       | Italic, Dotted UL    | `# PENDING: Waiting on API credentials from client`   |
| REVIEW        | Blue       | Bold, Underline      | `# REVIEW: Peer review this query for performance`    |
| REVIEW/REMOVE | Purple     | Bold, Dashed UL      | `# REVIEW/REMOVE: Check if legacy code can be removed` |
| ENHANCEMENT   | Green      | Bold, Wavy Underline | `# ENHANCEMENT: Add refresh button on data table`     |
| FIXME         | Orange-Red | Bold, Wavy Underline | `# FIXME: Memory leak under heavy load`               |
| TODO          | Sky Blue   | Underline            | `# TODO: Add unit tests for edge cases`               |
| REMOVE        | Purple     | Bold, Dashed UL      | `# REMOVE?: This block looks redundant`               |

---

Let me know if you'd like example screenshots, a markdown version, or help importing this into a team wiki or internal site!