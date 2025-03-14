# Sample LLM Agent: CongressRA
This repository contains code and resources for the paper "Agent-Enhanced Large Language Models for Researching Political Institutions" (2024) by Joseph Loffredo and Suyeol Yun from the department of political science at MIT. This code was written by Suyeol Yun, co-author of the paper.

## Congressional Gridlock Analysis

This project analyzes legislative gridlock in the U.S. Congress by examining policy issues and bill enactment rates. It uses AI-powered semantic clustering of New York Times articles and Congressional bills to identify policy areas and measure legislative productivity.

## Overview

The system performs the following main functions:

1. **Data Collection & Storage**: Processes and stores New York Times articles and Congressional bill data in vector databases
2. **Policy Clustering**: Uses AI to identify major policy areas from news coverage
3. **Bill Analysis**: Matches bills to policy clusters and calculates gridlock scores
4. **Gridlock Measurement**: Determines what percentage of policy areas had no successfully enacted legislation

## Components

### 1. Data Ingestion

#### NYT Articles Upserter
```
python nyt_upserter.py --batch-size 1000 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

- Processes approximately 800K New York Times articles
- Converts articles to vector embeddings
- Stores them in a vector database for semantic search

#### Bill Data Upserter
```
python bill_upserter.py --congress [CONGRESS_NUM] --batch-size 1000
```

- Processes approximately 80K Congressional bills
- Extracts metadata, text, and status information
- Converts to vector embeddings for semantic matching

### 2. Policy Cluster Analysis

```
python nyt_clusters.py --congress [CONGRESS_NUM] --output clusters_congress_[CONGRESS_NUM].json
```

- Analyzes NYT articles during a specific Congress
- Identifies major policy areas/issues using AI clustering
- Outputs clusters with representative articles and key terms
- Stores results in a JSON file for further analysis

### 3. Gridlock Score Calculation

```
python bill_analyzer.py --clusters-file clusters_congress_[CONGRESS_NUM].json --congress [CONGRESS_NUM] --output gridlock_score_congress_[CONGRESS_NUM].json
```

- Takes policy clusters as input
- Matches relevant bills to each policy cluster
- Calculates enactment rate for each cluster
- Computes overall gridlock score (percentage of policy areas with no enacted legislation)
- Outputs detailed results in JSON format

## Output Example

The system produces a JSON file (like `gridlock_score_congress_[CONGRESS_NUM].json`) containing:

- Overall gridlock statistics for the Congress
- Detailed breakdown of each policy cluster
- Bills associated with each cluster and their enactment status

## Usage

1. First, collect and process the NYT articles and bill data
2. Generate policy clusters for a specific Congress
3. Calculate gridlock scores based on the clusters

**Note**: This tool supports analysis for Congress numbers 113-118 (2013-2024).

## Requirements

- Python 3.7+
- Vector database (for semantic search)
- Access to New York Times article data
- Access to Congressional bill data

## Methodology

This project implements a modern approach to measuring Congressional gridlock, inspired by Binder (1999). It uses AI techniques to analyze which policy areas discussed in mainstream media fail to see successfully enacted legislation.

## Paper Abstract

The applications of Large Language Models (LLMs) in political science are rapidly expanding. This paper demonstrates how LLMs, when augmented with predefined functions and specialized tools, can serve as dynamic agents capable of streamlining tasks such as data collection, preprocessing, and analysis. Central to this approach is Agentic RAG, which equips LLMs with action-calling capabilities for interaction with external knowledge bases. Beyond information retrieval, LLM agents incorporate modular tools for tasks like document summarization, transcript coding, qualitative variable classification, and statistical modeling, enabling adaptability across diverse research contexts. To demonstrate the potential of this approach, we introduce CongressRA, an LLM agent designed to support scholars studying the U.S. Congress. Through this example, we highlight how LLM agents can reduce the costs of replicating, testing, and extending empirical research using the domain-specific data that drives the study of political institutions.
