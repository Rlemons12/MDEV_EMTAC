# modules/emtac_ai/training_scripts/dataset_gen/generate_parts_ner_train.py
import argparse
import json
import logging
import os
import re
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================
# Label space (final)
# =========================
# Permanently exclude CLASS / UD6 / TYPE
LABELS = [
    "O",
    "B-PART_NUMBER", "I-PART_NUMBER",
    "B-PART_NAME", "I-PART_NAME",
    "B-MANUFACTURER", "I-MANUFACTURER",
    "B-MODEL", "I-MODEL",
    ]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

# =========================
# Tokenization helpers
# =========================
TOKEN_RE = re.compile(r"\w+|\S", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def find_sublist_indices(tokens: List[str], sub: List[str]) -> Optional[Tuple[int, int]]:
    n, m = len(tokens), len(sub)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if tokens[i:i+m] == sub:
            return i, i+m
    return None


def _safe(row: pd.Series, col: str) -> str:
    v = row.get(col, "")
    if pd.isna(v):
        return ""
    return str(v).strip()

# =========================
# Natural-language augmentation (your patterns)
# =========================


PARTS_NATURAL_LANGUAGE_VARIATIONS = {

    # ITEMNUM (Internal Part Numbers) - Expanded from 15 to 45 variations
    "ITEMNUM": {
        "formal": [
            "part number {itemnum}",
            "item {itemnum}",
            "catalog number {itemnum}",
            "stock number {itemnum}",
            "SKU {itemnum}",
            "inventory number {itemnum}",
            "product code {itemnum}",
            "article number {itemnum}",
            "reference number {itemnum}",
            "item code {itemnum}",
            "catalog code {itemnum}",
            "stock code {itemnum}",
            "product number {itemnum}",
            "material number {itemnum}",
            "component number {itemnum}"
        ],
        "casual": [
            "part {itemnum}",
            "item# {itemnum}",
            "number {itemnum}",
            "code {itemnum}",
            "{itemnum}",
            "the {itemnum}",
            "#{itemnum}",
            "part# {itemnum}",
            "item {itemnum}",
            "piece {itemnum}",
            "thing {itemnum}",
            "that {itemnum}",
            "product {itemnum}",
            "stuff {itemnum}",
            "component {itemnum}"
        ],
        "contextual": [
            "the {itemnum} part",
            "anything with {itemnum}",
            "that {itemnum} item",
            "part called {itemnum}",
            "item labeled {itemnum}",
            "piece numbered {itemnum}",
            "component {itemnum}",
            "product marked {itemnum}",
            "item tagged {itemnum}",
            "part identified as {itemnum}",
            "article {itemnum}",
            "unit {itemnum}",
            "piece marked {itemnum}",
            "item referenced as {itemnum}",
            "component labeled {itemnum}"
        ]
    },

    # DESCRIPTION (Part Names) - Expanded from 15 to 45 variations
    "DESCRIPTION": {
        "formal": [
            "{description}",
            "a {description}",
            "the {description}",
            "{description} component",
            "{description} part",
            "{description} assembly",
            "{description} unit",
            "{description} piece",
            "{description} element",
            "{description} device",
            "{description} product",
            "{description} item",
            "{description} hardware",
            "{description} equipment",
            "{description} mechanism"
        ],
        "casual": [
            "some {description}",
            "any {description}",
            "that {description}",
            "{description} thing",
            "one of those {description}",
            "{description} stuff",
            "a {description} thingy",
            "{description} pieces",
            "{description} bits",
            "those {description} things",
            "{description} parts",
            "some {description} stuff",
            "{description} thingamajigs",
            "{description} whatsits",
            "{description} doohickeys"
        ],
        "contextual": [
            "something like a {description}",
            "type of {description}",
            "kind of {description}",
            "{description} or similar",
            "{description} style part",
            "{description} variant",
            "{description} or equivalent",
            "{description} replacement",
            "{description} substitute",
            "{description} alternative",
            "form of {description}",
            "{description} variety",
            "{description} version",
            "similar {description}",
            "comparable {description}"
        ]
    },

    # OEMMFG (Manufacturers) - Expanded from 15 to 45 variations
    "OEMMFG": {
        "formal": [
            "manufactured by {manufacturer}",
            "made by {manufacturer}",
            "from {manufacturer}",
            "{manufacturer} brand",
            "{manufacturer} manufactured",
            "produced by {manufacturer}",
            "{manufacturer} produced",
            "by {manufacturer}",
            "{manufacturer} company",
            "{manufacturer} corporation",
            "{manufacturer} products",
            "{manufacturer} branded",
            "{manufacturer} original",
            "genuine {manufacturer}",
            "authentic {manufacturer}"
        ],
        "casual": [
            "{manufacturer} stuff",
            "{manufacturer} parts",
            "any {manufacturer}",
            "{manufacturer} made",
            "some {manufacturer}",
            "{manufacturer} things",
            "{manufacturer} bits",
            "{manufacturer} gear",
            "{manufacturer} components",
            "that {manufacturer}",
            "{manufacturer} brand",
            "{manufacturer} items",
            "{manufacturer} products",
            "{manufacturer} pieces",
            "whatever {manufacturer} has"
        ],
        "contextual": [
            "something from {manufacturer}",
            "anything {manufacturer} makes",
            "whatever {manufacturer} has",
            "that {manufacturer} brand",
            "{manufacturer} or equivalent",
            "parts from {manufacturer}",
            "items by {manufacturer}",
            "products from {manufacturer}",
            "components from {manufacturer}",
            "anything made by {manufacturer}",
            "stuff from {manufacturer}",
            "parts manufactured by {manufacturer}",
            "equipment from {manufacturer}",
            "hardware from {manufacturer}",
            "devices from {manufacturer}"
        ]
    },

    # MODEL (Manufacturer's Part Numbers) - Expanded and corrected from 15 to 45 variations
    "MODEL": {
        "formal": [
            "OEM part {model}",
            "manufacturer part {model}",
            "OEM part number {model}",
            "manufacturer part number {model}",
            "original part {model}",
            "factory part {model}",
            "genuine part {model}",
            "OEM {model}",
            "manufacturer's part {model}",
            "original equipment part {model}",
            "factory part number {model}",
            "OEM reference {model}",
            "manufacturer reference {model}",
            "original part number {model}",
            "factory reference {model}"
        ],
        "casual": [
            "OEM {model}",
            "part {model}",
            "the {model}",
            "{model} part",
            "manufacturer {model}",
            "original {model}",
            "factory {model}",
            "genuine {model}",
            "that {model}",
            "{model} number",
            "OEM# {model}",
            "part# {model}",
            "mfg {model}",
            "orig {model}",
            "{model} piece"
        ],
        "contextual": [
            "something like OEM {model}",
            "similar to part {model}",
            "{model} or equivalent",
            "compatible with {model}",
            "replaces {model}",
            "substitute for {model}",
            "alternative to {model}",
            "replacement for {model}",
            "equivalent to {model}",
            "same as {model}",
            "matches {model}",
            "fits like {model}",
            "works like {model}",
            "functions as {model}",
            "performs like {model}"
        ]
    },

    # Additional contextual combinations for multi-entity queries
    "COMBINATIONS": {
        "formal": [
            "{description} manufactured by {manufacturer}",
            "{manufacturer} {description} part number {itemnum}",
            "OEM part {model} from {manufacturer}",
            "{description} with manufacturer part {model}",
            "{manufacturer} original {description}",
            "genuine {manufacturer} {description} part {itemnum}",
            "{description} component from {manufacturer}",
            "authentic {manufacturer} part {itemnum}",
            "original {manufacturer} {description} with OEM {model}",
            "factory {manufacturer} {description}"
        ],
        "casual": [
            "{manufacturer} {description} stuff",
            "that {description} from {manufacturer}",
            "{manufacturer} part {itemnum}",
            "some {description} with OEM {model}",
            "{manufacturer} {description} thing",
            "any {description} from {manufacturer}",
            "{manufacturer} {description} bits",
            "that {itemnum} part",
            "{description} with part {model}",
            "{manufacturer} {description} pieces"
        ],
        "contextual": [
            "something like {description} from {manufacturer}",
            "type of {description} with OEM {model}",
            "{description} or similar from {manufacturer}",
            "equivalent {description} part {itemnum}",
            "replacement {description} from {manufacturer}",
            "substitute for {manufacturer} {description}",
            "alternative to part {itemnum}",
            "compatible {description} with OEM {model}",
            "similar {manufacturer} {description}",
            "comparable {description} part {itemnum}"
        ]
    },

    # Technical/Industry specific variations
    "TECHNICAL": {
        "formal": [
            "OEM specification {model}",
            "manufacturer specification {itemnum}",
            "technical specification {description}",
            "engineering part {model}",
            "certified {manufacturer} component",
            "approved {description} from {manufacturer}",
            "specification compliant {description}",
            "standard {manufacturer} part {itemnum}",
            "regulatory approved {description}",
            "quality assured {manufacturer} {description}"
        ],
        "casual": [
            "spec {model}",
            "standard {description}",
            "approved {manufacturer} stuff",
            "certified {description}",
            "quality {manufacturer} parts",
            "legit {description}",
            "proper {manufacturer} {description}",
            "real {description}",
            "good {manufacturer} parts",
            "right {description} stuff"
        ]
    },

    # Urgency/Priority variations
    "URGENCY": {
        "formal": [
            "urgently need {description}",
            "immediately require {itemnum}",
            "priority order for {manufacturer} {description}",
            "emergency replacement {description}",
            "critical need for part {itemnum}",
            "expedited request for {description}",
            "rush order {manufacturer} part {model}",
            "urgent requirement for {description}",
            "immediate need for {itemnum}",
            "emergency order {description}"
        ],
        "casual": [
            "really need {description}",
            "gotta have {itemnum}",
            "need {description} ASAP",
            "must have {manufacturer} {description}",
            "desperate for {description}",
            "really need that {itemnum}",
            "gotta get {description}",
            "need {manufacturer} stuff now",
            "must find {description}",
            "really want {itemnum}"
        ]
    }
}

PARTS_ENHANCED_QUERY_TEMPLATES = [
    # Single entity queries - Part Numbers (20)
    "PN {itemnum}",
    "P/N {itemnum}",
    "part no {itemnum}",
    "part # {itemnum}",
    "pn {itemnum}",
    "p/n {itemnum}",
    "partno {itemnum}",
    "pn: {itemnum}",
    "p/n: {itemnum}",
    "I need {itemnum_formal}",
    "Do you stock {itemnum_casual}?",
    "I require {itemnum_formal}",
    "Do you carry {itemnum_contextual}?",
    "Looking for {itemnum_casual}",
    "Can I get {itemnum_formal}?",
    "Do you have {itemnum_casual} in stock?",
    "I'm looking for {itemnum_contextual}",
    "Need {itemnum_casual}",
    "What about {itemnum_formal}?",
    "Can you find {itemnum_casual}?",
    "I want {itemnum_formal}",
    "Is {itemnum_casual} available?",
    "Do you sell {itemnum_formal}?",
    "I'm interested in {itemnum_casual}",
    "Can I order {itemnum_formal}?",
    "Do you have {itemnum_contextual} on hand?",
    "I need {itemnum_casual}",
    "Looking for {itemnum_formal}",
    "Show me {itemnum_casual}",

    # Single entity queries - Descriptions (20)
    "Do you have {description_formal}?",
    "I need {description_casual}",
    "I'm searching for {description_formal}",
    "Show me {description_casual}",
    "I need {description_formal}",
    "Do you carry {description_casual}?",
    "Looking for {description_formal}",
    "I require {description_casual}",
    "Can I get {description_formal}?",
    "Do you stock {description_casual}?",
    "I want {description_formal}",
    "Any {description_casual} available?",
    "I'm looking for {description_contextual}",
    "Can you find {description_formal}?",
    "Do you sell {description_casual}?",
    "I'm interested in {description_contextual}",
    "Need {description_casual}",
    "What about {description_formal}?",
    "Can I order {description_casual}?",
    "Find me {description_contextual}",

    # Single entity queries - Manufacturers (20)
    "I'm looking for {manufacturer_contextual}",
    "Looking for {manufacturer_casual}",
    "Need {manufacturer_contextual}",
    "Find {manufacturer_casual}",
    "Do you carry {manufacturer_formal}?",
    "I need {manufacturer_casual}",
    "Any {manufacturer_contextual} available?",
    "I'm searching for {manufacturer_formal}",
    "Do you stock {manufacturer_casual}?",
    "Show me {manufacturer_formal}",
    "I want {manufacturer_casual}",
    "Can you find {manufacturer_contextual}?",
    "Do you have {manufacturer_formal}?",
    "I'm interested in {manufacturer_casual}",
    "Looking for {manufacturer_formal}",
    "Need {manufacturer_casual}",
    "Any {manufacturer_contextual}?",
    "I require {manufacturer_formal}",
    "What {manufacturer_casual} do you have?",
    "Find me {manufacturer_contextual}",

    # Single entity queries - Manufacturer Part Numbers (20)
    "Can I get {model_formal}?",
    "Can you find {model_casual}?",
    "Any {model_formal} available?",
    "I need {model_casual}",
    "Do you have {model_formal}?",
    "Looking for {model_casual}",
    "I'm searching for {model_formal}",
    "Show me {model_casual}",
    "I want {model_formal}",
    "Do you stock {model_casual}?",
    "I require {model_formal}",
    "Can I order {model_casual}?",
    "Is {model_formal} available?",
    "Do you carry {model_casual}?",
    "I'm looking for {model_contextual}",
    "Need {model_casual}",
    "Find {model_formal}",
    "What about {model_casual}?",
    "Can you get {model_contextual}?",
    "I'm interested in {model_formal}",
    "This is the OEM number{model_formal}",
    "manufacturer part number {model_formal}",


    # Two entity combinations - Description + Manufacturer (20)
    "I need {description_formal} from {manufacturer_formal}",
    "Do you have {description_casual} by {manufacturer_casual}?",
    "I'm looking for {manufacturer_formal} {description_casual}",
    "I need {manufacturer_casual} {description_formal}",
    "Looking for {description_contextual} made by {manufacturer_formal}",
    "Do you carry {description_formal}, {manufacturer_casual}?",
    "Need {description_casual} from {manufacturer_contextual}",
    "Any {manufacturer_formal} {description_casual} available?",
    "I want {description_formal} by {manufacturer_casual}",
    "Do you stock {manufacturer_contextual} {description_formal}?",
    "I'm searching for {description_casual} from {manufacturer_formal}",
    "Can I get {manufacturer_casual} {description_contextual}?",
    "Show me {description_formal} made by {manufacturer_casual}",
    "I require {manufacturer_formal} {description_casual}",
    "Find {description_contextual} from {manufacturer_formal}",
    "Do you have {manufacturer_casual} {description_formal}?",
    "I'm looking for {description_casual}, {manufacturer_contextual}",
    "Need some {manufacturer_formal} {description_casual}",
    "What {description_contextual} does {manufacturer_casual} make?",
    "Can you find {manufacturer_formal} {description_casual}?",

    # Two entity combinations - Description + Manufacturer Part Number (20)
    "Can I get {description_formal} with {model_formal}?",
    "Show me {description_casual}, {model_casual}",
    "I need {description_formal} with {model_casual}",
    "Do you have {description_casual}, {model_formal}?",
    "Looking for {description_contextual} with {model_casual}",
    "I want {description_formal} with {model_contextual}",
    "Any {description_casual} with {model_formal} available?",
    "I'm searching for {description_formal}, {model_casual}",
    "Do you stock {description_casual} with {model_contextual}?",
    "Can you find {description_contextual} with {model_formal}?",
    "I require {description_formal} with {model_casual}",
    "Need {description_casual} with {model_formal}",
    "Find {description_contextual}, {model_casual}",
    "Do you carry {description_formal} with {model_contextual}?",
    "I'm looking for {description_casual} with {model_formal}",
    "What about {description_contextual} with {model_casual}?",
    "Can I order {description_formal} with {model_contextual}?",
    "Is {description_casual} with {model_formal} available?",
    "Show me {description_contextual} with {model_casual}",
    "I'm interested in {description_formal} with {model_contextual}",

    # Two entity combinations - Part Number + Manufacturer (20)
    "Do you stock {itemnum_formal} from {manufacturer_formal}?",
    "Find {itemnum_casual} made by {manufacturer_casual}",
    "I need {manufacturer_formal} {itemnum_casual}",
    "Do you have {itemnum_contextual} by {manufacturer_formal}?",
    "Looking for {itemnum_formal} from {manufacturer_casual}",
    "Can I get {manufacturer_contextual} {itemnum_formal}?",
    "I want {itemnum_casual} made by {manufacturer_formal}",
    "Do you carry {manufacturer_casual} {itemnum_contextual}?",
    "I'm searching for {itemnum_formal} from {manufacturer_casual}",
    "Show me {manufacturer_contextual} {itemnum_casual}",
    "Any {itemnum_formal} from {manufacturer_casual} available?",
    "I require {manufacturer_formal} {itemnum_contextual}",
    "Need {itemnum_casual} by {manufacturer_formal}",
    "Find {manufacturer_casual} {itemnum_contextual}",
    "Do you stock {manufacturer_contextual} {itemnum_formal}?",
    "I'm looking for {itemnum_casual}, {manufacturer_formal}",
    "Can you find {itemnum_contextual} from {manufacturer_casual}?",
    "What about {manufacturer_formal} {itemnum_casual}?",
    "Is {itemnum_contextual} from {manufacturer_formal} available?",
    "I'm interested in {manufacturer_casual} {itemnum_contextual}",

    # Two entity combinations - Part Number + Manufacturer Part Number (15)
    "I'm searching for {itemnum_formal} or {model_casual}",
    "Do you have {itemnum_casual} or {model_formal}?",
    "Looking for {itemnum_contextual}, also {model_casual}",
    "I need {itemnum_formal} which is {model_contextual}",
    "Can I get {itemnum_casual} or {model_formal}?",
    "Find {itemnum_contextual}, that's {model_casual}",
    "Do you stock {itemnum_formal} or {model_contextual}?",
    "I want {itemnum_casual}, {model_formal}",
    "Any {itemnum_contextual} or {model_casual} available?",
    "Show me {itemnum_formal} or {model_contextual}",
    "I require {itemnum_casual} or {model_formal}",
    "Need {itemnum_contextual}, {model_casual}",
    "What about {itemnum_formal} or {model_contextual}?",
    "Can you find {itemnum_casual} or {model_formal}?",
    "I'm looking for {itemnum_contextual} or {model_casual}",

    # Two entity combinations - Manufacturer + Manufacturer Part Number (15)
    "Can you find {manufacturer_formal} {model_casual}?",
    "I require {model_formal} from {manufacturer_casual}",
    "Do you have {manufacturer_contextual} {model_formal}?",
    "Looking for {manufacturer_formal} {model_casual}",
    "I need {manufacturer_casual} {model_contextual}",
    "Can I get {model_formal} from {manufacturer_casual}?",
    "Do you stock {manufacturer_contextual} {model_formal}?",
    "I want {manufacturer_formal} {model_casual}",
    "Any {manufacturer_casual} {model_contextual} available?",
    "Show me {manufacturer_formal} {model_casual}",
    "I'm searching for {manufacturer_contextual} {model_formal}",
    "Find {manufacturer_formal} {model_casual}",
    "Do you carry {manufacturer_casual} {model_contextual}?",
    "I'm looking for {manufacturer_formal} {model_casual}",
    "Need {manufacturer_contextual} {model_formal}",

    # Three entity combinations - Description + Manufacturer + Manufacturer Part Number (25)
    "I need {description_formal} from {manufacturer_formal}, {model_casual}",
    "Do you have {manufacturer_casual} {description_formal} with {model_contextual}?",
    "Looking for {description_contextual} made by {manufacturer_formal}, {model_casual}",
    "Can I get {description_formal} by {manufacturer_casual}, {model_contextual}?",
    "I need {manufacturer_formal} {description_casual}, {model_formal}",
    "Any {manufacturer_contextual} {description_formal} with {model_casual} available?",
    "Need {description_casual} from {manufacturer_formal}, {model_contextual}",
    "I'm searching for {description_contextual}, {manufacturer_casual}, {model_formal}",
    "Do you stock {manufacturer_formal} {description_casual} with {model_contextual}?",
    "Can you find {description_formal} from {manufacturer_contextual}, {model_casual}?",
    "I want {manufacturer_casual} {description_contextual} with {model_formal}",
    "Show me {description_formal} by {manufacturer_casual}, {model_contextual}",
    "I require {description_contextual} from {manufacturer_formal}, {model_casual}",
    "Find {manufacturer_casual} {description_formal} with {model_contextual}",
    "Do you carry {description_casual}, {manufacturer_contextual}, {model_formal}?",
    "I'm looking for {manufacturer_formal} {description_contextual}, {model_casual}",
    "What about {description_casual} from {manufacturer_formal}, {model_contextual}?",
    "Can I order {manufacturer_contextual} {description_formal} with {model_casual}?",
    "Is {description_contextual} by {manufacturer_casual}, {model_formal} available?",
    "I'm interested in {manufacturer_formal} {description_casual} with {model_contextual}",
    "Need some {description_contextual} from {manufacturer_casual}, {model_formal}",
    "Do you have {description_formal}, {manufacturer_contextual}, {model_casual}?",
    "Looking for {manufacturer_casual} {description_contextual}, {model_formal}",
    "Can you get {description_formal} from {manufacturer_casual}, {model_contextual}?",
    "Find me {manufacturer_contextual} {description_casual} with {model_formal}",

    # Three entity combinations - Part Number + Description + Manufacturer (25)
    "I'm looking for {itemnum_formal}, {description_casual} from {manufacturer_formal}",
    "Do you stock {itemnum_contextual} which is {description_formal} from {manufacturer_casual}?",
    "I need {manufacturer_formal} {description_contextual}, {itemnum_casual}",
    "Can you find {itemnum_formal}, that's the {description_casual} from {manufacturer_contextual}?",
    "Do you carry {manufacturer_casual} {itemnum_contextual}, the {description_formal}?",
    "I require {itemnum_casual}, {description_contextual} manufactured by {manufacturer_formal}",
    "Looking for {itemnum_contextual}, a {description_formal} from {manufacturer_casual}",
    "Do you have {itemnum_formal}, which is {manufacturer_contextual} {description_casual}?",
    "I need {description_formal} from {manufacturer_casual}, {itemnum_contextual}",
    "Can I get {itemnum_casual}, the {manufacturer_formal} {description_contextual}?",
    "Find {manufacturer_contextual} {description_casual}, {itemnum_formal}",
    "Do you stock {description_contextual} from {manufacturer_formal}, {itemnum_casual}?",
    "I want {itemnum_formal}, {description_casual} by {manufacturer_contextual}",
    "Any {manufacturer_formal} {description_contextual}, {itemnum_casual} available?",
    "Show me {itemnum_contextual}, {description_formal} from {manufacturer_casual}",
    "I'm searching for {description_casual}, {itemnum_formal} from {manufacturer_contextual}",
    "Need {manufacturer_casual} {itemnum_contextual}, the {description_formal}",
    "What about {itemnum_formal}, {manufacturer_contextual} {description_casual}?",
    "Can you find {description_contextual} from {manufacturer_formal}, {itemnum_casual}?",
    "I'm looking for {manufacturer_casual} {description_formal}, {itemnum_contextual}",
    "Do you have {description_casual}, {manufacturer_contextual} {itemnum_formal}?",
    "I require {itemnum_casual}, which is {description_contextual} from {manufacturer_formal}",
    "Find {description_formal} {itemnum_contextual} from {manufacturer_casual}",
    "Can I order {manufacturer_contextual} {description_casual}, {itemnum_formal}?",
    "Is {itemnum_contextual}, {description_formal} from {manufacturer_casual} available?",

    # Casual/Conversational queries (15)
    "Got any {description_casual}?",
    "What {manufacturer_casual} you got?",
    "You have {itemnum_casual}?",
    "Any {description_contextual} in stock?",
    "You guys carry {manufacturer_casual}?",
    "What about that {model_casual}?",
    "You got {description_formal} from {manufacturer_casual}?",
    "Do ya have {itemnum_contextual}?",
    "Any {manufacturer_formal} {description_casual}?",
    "You stock {model_contextual}?",
    "Got {description_casual}?",
    "You carry {itemnum_formal}?",
    "What {description_contextual} you got?",
    "You have {manufacturer_casual}?",
    "Any {model_formal} available?",

    "Do you have PN {itemnum_formal}?",
    "P/N {itemnum_casual}",
    "part no. {itemnum_formal}",
    "cat no {itemnum_casual}",
    "sku {itemnum_formal}",
    "Check {itemnum_casual} (part number)",

    # --- Part number in context (questions about name / maker / cross) ---
    "What's the name for {itemnum_formal}?",
    "Who makes {itemnum_casual}?",
    "Cross for part {itemnum_formal}?",
    "Is {itemnum_casual} the same as {model_formal}?",
    "Need the manufacturer for {itemnum_formal}",

    # --- Manufacturer synonyms/brand phrasing ---
    "{manufacturer_formal} brand",
    "OEM {manufacturer_casual} parts",
    "by {manufacturer_formal}",
    "made by {manufacturer_casual}",
    "from {manufacturer_formal}",

    # --- Descriptions w/ measurements & variants (common tokenization pain) ---
    "{description_formal} 1/2\"",
    "{description_casual} 0-160 PSI",
    "{description_formal} NPT",
    "stainless {description_casual}",
    "{description_contextual} backmount gauge 0–160 psi",

    # --- Mixed signals (PN + model + mfr) in real orderings ---
    "{itemnum_formal} – {manufacturer_formal} – {model_casual}",
    "{manufacturer_casual} {model_formal} ({itemnum_casual})",
    "Need {description_formal}, {manufacturer_casual}, MPN {model_formal}, PN {itemnum_casual}",

    # --- Imperatives & shortforms (chatty) ---
    "show {itemnum_casual}",
    "pull {manufacturer_casual} {description_formal}",
    "quote {model_formal}",
    "price {itemnum_casual}",

    # --- Availability & quantity (ensure surrounding words don’t confuse NER) ---
    "QTY 4 of {itemnum_formal}?",
    "lead time on {model_casual} by {manufacturer_formal}?",
    "in stock: {description_formal} from {manufacturer_casual}?",

    # --- Punctuation / wrappers / noise (robustness) ---
    "[{itemnum_formal}]",
    "\"{model_formal}\" from {manufacturer_casual}",
    "re: {description_formal}",
    "FYI {itemnum_casual} – urgent",
    "(need) {manufacturer_formal} {description_casual}",

    # --- Negations & clarifications (don’t trick the tagger) ---
    "Not {model_casual}, need {itemnum_formal}",
    "Looking for {manufacturer_formal}, not {manufacturer_casual}",
    "Need {description_formal}, not the {description_contextual}",


    # --- Near-real typos/spaces/hyphens around entities (tokenization stress) ---
    "PN:{itemnum_formal}",
    "MPN:{model_formal}",
    "part# {itemnum_casual}",
    "model# {model_casual}",
    "{manufacturer_formal}/{description_casual}",

    # Technical/Specific queries (15)
    "I need replacement {description_formal}",
    "Looking for {manufacturer_formal}",
    "Do you have genuine {manufacturer_casual} {description_formal}?",
    "I need {description_contextual} with {model_formal}",
    "Looking for {description_formal} that uses {model_casual}",
    "Do you have {manufacturer_contextual} original parts?",
    "I need authentic {manufacturer_formal} {description_casual}",
    "Looking for {description_contextual} for {model_formal}",
    "Do you stock {itemnum_formal}?",
    "I need factory {manufacturer_casual} {description_formal}",
    "Looking for genuine {itemnum_contextual}",
    "Do you have original {manufacturer_formal} {model_casual}?",
    "I need {description_formal} that works with {model_contextual}",
    "Looking for {manufacturer_casual} certified {description_formal}",
    "Do you have {description_contextual} for {model_formal}?"
]

# =========================
# NL augmentation helpers
# =========================


def sample_slot_variations(itemnum: str, description: str, manufacturer: str, model: str, rng: random.Random) -> Dict:
    def pick(slot_key, **fmt):
        return {
            "formal": rng.choice(PARTS_NATURAL_LANGUAGE_VARIATIONS[slot_key]["formal"]).format(**fmt),
            "casual": rng.choice(PARTS_NATURAL_LANGUAGE_VARIATIONS[slot_key]["casual"]).format(**fmt),
            "contextual": rng.choice(PARTS_NATURAL_LANGUAGE_VARIATIONS[slot_key]["contextual"]).format(**fmt),
        }
    return {
        "itemnum":      pick("ITEMNUM",      itemnum=itemnum),
        "description":  pick("DESCRIPTION",  description=description),
        "manufacturer": pick("OEMMFG",       manufacturer=manufacturer),
        "model":        pick("MODEL",        model=model),
    }



def build_augmented_sentence(row: pd.Series, template: str, rng: random.Random) -> Tuple[str, Dict[str, str]]:
    # Pull raw values
    itemnum      = _safe(row, "ITEMNUM")
    description  = _safe(row, "DESCRIPTION")
    manufacturer = _safe(row, "OEMMFG")
    model        = _safe(row, "MODEL")

    # Surface normalization for more natural queries (only for the *variants*)
    desc_surf = description.lower() if description else ""
    mfg_surf  = manufacturer.lower() if manufacturer else ""

    pick = rng.choice

    fmt = {
        # RAW keys (for templates like "PN {itemnum}")
        "itemnum": itemnum,
        "description": description,
        "manufacturer": manufacturer,
        "model": model,

        # VARIANTS
        "itemnum_formal":       pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["ITEMNUM"]["formal"]).format(itemnum=itemnum) if itemnum else "",
        "itemnum_casual":       pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["ITEMNUM"]["casual"]).format(itemnum=itemnum) if itemnum else "",
        "itemnum_contextual":   pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["ITEMNUM"]["contextual"]).format(itemnum=itemnum) if itemnum else "",

        "description_formal":     pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["DESCRIPTION"]["formal"]).format(description=desc_surf) if desc_surf else "",
        "description_casual":     pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["DESCRIPTION"]["casual"]).format(description=desc_surf) if desc_surf else "",
        "description_contextual": pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["DESCRIPTION"]["contextual"]).format(description=desc_surf) if desc_surf else "",

        "manufacturer_formal":     pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["OEMMFG"]["formal"]).format(manufacturer=mfg_surf) if mfg_surf else "",
        "manufacturer_casual":     pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["OEMMFG"]["casual"]).format(manufacturer=mfg_surf) if mfg_surf else "",
        "manufacturer_contextual": pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["OEMMFG"]["contextual"]).format(manufacturer=mfg_surf) if mfg_surf else "",

        "model_formal":     pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["MODEL"]["formal"]).format(model=model) if model else "",
        "model_casual":     pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["MODEL"]["casual"]).format(model=model) if model else "",
        "model_contextual": pick(PARTS_NATURAL_LANGUAGE_VARIATIONS["MODEL"]["contextual"]).format(model=model) if model else "",
    }

    sentence = template.format(**fmt)

    # Entity map used for tagging (stick to canonical/raw values)
    ent_surface: Dict[str, str] = {}
    if itemnum:
        ent_surface["PART_NUMBER"] = itemnum
    if description:
        ent_surface["PART_NAME"] = desc_surf
    if manufacturer:
        ent_surface["MANUFACTURER"] = mfg_surf
    if model:
        ent_surface["MODEL"] = model

    return sentence, ent_surface


def tag_tokens_surface(sentence: str, entity_surface_map: Dict[str, str]) -> Tuple[List[str], List[int]]:
    tokens = tokenize(sentence)
    tags = ["O"] * len(tokens)
    for ent, surface in entity_surface_map.items():
        if not surface:
            continue
        ent_tokens = tokenize(surface)
        span = find_sublist_indices(tokens, ent_tokens)
        if not span:
            continue
        s, e = span
        tags[s] = f"B-{ent}"
        for i in range(s+1, e):
            tags[i] = f"I-{ent}"
    ner_ids = [LABEL2ID.get(t, 0) for t in tags]
    return tokens, ner_ids

def build_row_sentence(row: pd.Series) -> str:
    pn  = _safe(row, "ITEMNUM")
    nm  = _safe(row, "DESCRIPTION")
    mfg = _safe(row, "OEMMFG")
    mdl = _safe(row, "MODEL")

    parts = []
    if pn:  parts.append(f"part number {pn}")
    if nm:  parts.append(f"name {nm}")
    if mfg: parts.append(f"made by {mfg}")
    if mdl: parts.append(f"model {mdl}")

    return "This is " + ", ".join(parts) + "." if parts else "This is a part."


def build_row_sentence_alt(row: pd.Series) -> str:
    pn  = _safe(row, "ITEMNUM")
    nm  = _safe(row, "DESCRIPTION")
    mfg = _safe(row, "OEMMFG")
    mdl = _safe(row, "MODEL")

    primary = []
    if nm:  primary.append(f"name {nm}")
    if pn:  primary.append(f"part number {pn}")
    if mdl: primary.append(f"model {mdl}")
    if mfg: primary.append(f"made by {mfg}")

    return "Part details: " + ", ".join(primary) + "." if primary else "Part details unavailable."


# =========================
# Column→entity map (aligned with labels)
# =========================
DEFAULT_COLMAP: Dict[str, str] = {
    "ITEMNUM": "PART_NUMBER",
    "DESCRIPTION": "PART_NAME",
    "OEMMFG": "MANUFACTURER",
    "MODEL": "MODEL",

}

# =========================
# JSONL writer
# =========================
def write_example(f, sentence: str, ent_surface_map: Dict[str, str]):
    tokens, ner_ids = tag_tokens_surface(sentence, ent_surface_map)
    f.write(json.dumps({"tokens": tokens, "ner_tags": ner_ids}, ensure_ascii=False) + "\n")


def generate_ner_training_file(
    excel_path: str,
    out_hf_path: str,
    sheet_name: Optional[str] = None,
    augment_basic: bool = False,
    augment_nl: bool = False,
    seed: int = 42,
) -> int:
    rng = random.Random(seed)
    logging.info("Reading Excel: %s", excel_path)
    df = pd.read_excel(excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]

    os.makedirs(os.path.dirname(out_hf_path), exist_ok=True)
    rows_written = 0
    with open(out_hf_path, "w", encoding="utf-8") as f_hf:
        for _, row in df.iterrows():
            # --- NL augmentation path ---
            if augment_nl:
                tmpl = rng.choice(PARTS_ENHANCED_QUERY_TEMPLATES)
                sentence, ent_map = build_augmented_sentence(row, tmpl, rng)
                write_example(f_hf, sentence, ent_map)
                rows_written += 1
                # Also add a second variant (optional) for more variety
                if augment_basic:
                    # basic alt too
                    sentence2 = build_row_sentence_alt(row)
                    ent_map2 = {
                        "PART_NUMBER": _safe(row, "ITEMNUM"),
                        "PART_NAME": _safe(row, "DESCRIPTION").lower(),
                        "MANUFACTURER": _safe(row, "OEMMFG").lower(),
                        "MODEL": _safe(row, "MODEL"),

                    }
                    # remove empty keys
                    ent_map2 = {k: v for k, v in ent_map2.items() if v}
                    write_example(f_hf, sentence2, ent_map2)
                    rows_written += 1
            else:
                # --- basic path (original patterns) ---
                sentence = build_row_sentence(row)
                ent_map = {
                    "PART_NUMBER": _safe(row, "ITEMNUM"),
                    "PART_NAME": _safe(row, "DESCRIPTION"),
                    "MANUFACTURER": _safe(row, "OEMMFG"),
                    "MODEL": _safe(row, "MODEL"),

                }
                # Lowercase only name/mfg in place if you want to match basic builders exactly:
                if ent_map["PART_NAME"]:
                    ent_map["PART_NAME"] = ent_map["PART_NAME"]
                if ent_map["MANUFACTURER"]:
                    ent_map["MANUFACTURER"] = ent_map["MANUFACTURER"]
                ent_map = {k: v for k, v in ent_map.items() if v}
                write_example(f_hf, sentence, ent_map)
                rows_written += 1

                if augment_basic:
                    sentence2 = build_row_sentence_alt(row)
                    ent_map2 = {
                        "PART_NUMBER": _safe(row, "ITEMNUM"),
                        "PART_NAME": _safe(row, "DESCRIPTION"),
                        "MANUFACTURER": _safe(row, "OEMMFG"),
                        "MODEL": _safe(row, "MODEL"),

                    }
                    ent_map2 = {k: v for k, v in ent_map2.items() if v}
                    write_example(f_hf, sentence2, ent_map2)
                    rows_written += 1

    logging.info("Wrote HF token/BIO JSONL: %s (%d rows)", out_hf_path, rows_written)
    return rows_written

# =========================
# CLI + interactive prompts
# =========================


def parse_args() -> argparse.Namespace:
    # Import config here to avoid circular imports during package init
    from modules.emtac_ai import config
    default_excel = os.path.join(config.ORC_TRAINING_DATA_LOADSHEET, "parts_loadsheet.xlsx")

    p = argparse.ArgumentParser(description="Generate Parts NER training data from Excel.")
    p.add_argument("--excel", required=False, default=default_excel,
                   help=f"Path to 'Copy of MP2_ITEMS_BOMS.xlsx'. Default: {default_excel}")
    p.add_argument("--sheet", default=None, help="Worksheet name (optional).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    ans = input(prompt + suffix).strip().lower()
    if ans == "" and default:
        return True
    return ans == "y"


def main():
    # Import config inside main as well (keeps module import side-effect free)
    from modules.emtac_ai import config
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(message)s")

    print("\n=== Parts NER Dataset Generator ===")
    use_aug_nl = ask_yes_no("Use natural-language augmentation?", default=True)
    use_aug_basic = False
    if not use_aug_nl:
        use_aug_basic = ask_yes_no("Use simple second-phrase augmentation?", default=True)

    out_hf = os.path.join(config.ORC_PARTS_TRAIN_DATA_DIR, "ner_train_parts.jsonl")
    os.makedirs(os.path.dirname(out_hf), exist_ok=True)

    logging.info("Excel   : %s", args.excel)
    logging.info("HF out  : %s", out_hf)
    logging.info("Aug NL  : %s", use_aug_nl)
    logging.info("Aug basic: %s", use_aug_basic)

    generate_ner_training_file(
        excel_path=args.excel,
        sheet_name=args.sheet,
        out_hf_path=out_hf,
        augment_basic=use_aug_basic,
        augment_nl=use_aug_nl,
        seed=42,
    )

if __name__ == "__main__":
    main()
