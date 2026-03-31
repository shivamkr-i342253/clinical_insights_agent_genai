# Workflow Changes: General Query Handling

## Overview
Modified the workflow to handle general queries (non-clinical) by terminating the workflow after the Assistant node instead of proceeding through Data Cleanup and subsequent analysis nodes.

## Changes Made

### 1. Enhanced Router Function: `route_after_assistant()`
**Location**: `graph/workflow.py` (lines ~40-75)

**Changes**:
- Updated return type from `Literal["data_loader", "data_cleanup"]` to `Literal["data_loader", "data_cleanup", "__end__"]`
- Added logic to detect general queries
- Routes general queries to `__end__` (END node) to terminate workflow early

**Logic Flow**:
1. Check if LLM made tool calls → route to `data_loader` (unchanged)
2. Check if response is a general query → route to `__end__` (NEW)
3. Default to `data_cleanup` for clinical queries (unchanged)

### 2. New Helper Function: `_is_general_query()`
**Location**: `graph/workflow.py` (lines ~76-128)

**Purpose**: Intelligently detect if the assistant's response is answering a general query

**Detection Criteria**:
- **Contains clinical keywords**: Returns False if response mentions "clinical trial", "patient data", "drug efficacy", etc.
- **Requests clinical data**: Returns False if response asks for data/uploads
- **Substantial response**: Returns True if response is >20 characters and not clinical-related

**Clinical Keywords Detected**:
- clinical trial, patient data, medical records
- drug efficacy, safety profile, adverse event
- trial data, cohort, compliance, dosage
- regulatory, fda, treatment outcome, diagnosis, prescription
- check the data, analyze the data, review the trial, examine clinical

### 3. Updated Graph Edges
**Location**: `graph/workflow.py` (lines ~267-281)

**Changes**:
```python
# Before:
graph.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "data_loader": "data_loader",
        "data_cleanup": "data_cleanup",
    },
)

# After:
graph.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "data_loader": "data_loader",
        "data_cleanup": "data_cleanup",
        "__end__": END,  # General query - end workflow and return answer
    },
)
```

## Behavior Examples

### General Query (Terminates at Assistant Node)
**Input**: "What's the largest river in the world?"
**Response**: Direct answer from LLM
**Nodes Executed**: Assistant → END
**Remaining Nodes Skipped**: Data Cleanup, Call Subgraphs, Trial Insights, Cohort Comparison, etc.

### Clinical Query (Continues Through Full Workflow)
**Input**: "Analyze patient compliance in trial data"  
**Response**: Request for clinical trial data
**Nodes Executed**: Assistant → Data Cleanup → Call Subgraphs → Trial Insights → ... (full pipeline)

## Benefits

1. **Efficiency**: General queries are answered immediately without unnecessary data processing
2. **Reduced Latency**: No need to load clinical data or run subgraphs for non-clinical questions
3. **Cost Savings**: Avoids expensive operations on unrelated queries
4. **Better UX**: Users get immediate answers to general questions
5. **Maintains Functionality**: Clinical queries still go through full analysis pipeline

## Testing Recommendations

Test the following scenarios:
1. ✅ General knowledge query: "What's the capital of France?"
2. ✅ General knowledge query: "How many planets are in our solar system?"
3. ✅ Clinical query: "What's the efficacy of drug X?"
4. ✅ Clinical query asking for data: "Load the trial data from the CSV"
5. ✅ Ambiguous query: Verify it routes correctly based on response context

## Rollback Instructions

If needed, revert changes:
1. Remove the `_is_general_query()` function
2. Restore original `route_after_assistant()` with return type `Literal["data_loader", "data_cleanup"]`
3. Remove `"__end__": END,` from the conditional edges mapping
