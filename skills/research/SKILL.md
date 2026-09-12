---
name: research
description: >
  Conducts comprehensive web research on any topic using search APIs 
  and web scraping. Use when the user asks for research, information 
  gathering, competitive analysis, or needs current data from the web.
license: MIT
compatibility: Requires internet access and Tavily API key
allowed-tools: internet_search Read Write
---

# Web Research Skill

## Overview
This skill provides structured web research capabilities. It handles breaking down complex research questions into targeted search queries and synthesizing findings.

## Instructions
1. Parse the research query to identify key topics and missing information.
2. Use the `internet_search` tool to gather initial results (results are ~300-char snippets).
3. For pages that matter, use `fetch_url` to get the full page content before summarizing them.
4. If results are insufficient, try alternative search queries.
5. Synthesize the findings into a clear, structured summary, prioritizing factual accuracy and depth.
6. Check what already exists in `./workspace` with `list_files` (and `search_files` to locate a specific note), then save your findings as a markdown file there — never guess a path that may already be taken by another subagent.

## Completion
End with a final markdown summary containing:
- A comprehensive summary of the findings
- Key facts and findings as a bulleted list
- The sources used
- A confidence score (0-1) in the findings

## Examples
### Input
"Research the latest developments in quantum computing"

### Expected Behavior
The agent should conduct 3-5 targeted searches, summarize the key breakthroughs (e.g., error correction, new qubit types), and write a report to `workspace/quantum_research.md`.
