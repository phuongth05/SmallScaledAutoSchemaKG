from tqdm import tqdm
import random
import logging
import csv
import os
import hashlib
import re
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml import get_node_id
from atlas_rag.llm_generator.prompt.triple_extraction_prompt import CONCEPT_INSTRUCTIONS
import pickle
import json
from datetime import datetime, timezone
from pathlib import Path

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Increase the field size limit
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB limit



def build_batch_data(sessions, batch_size):
    batched_sessions = []
    for i in range(0, len(sessions), batch_size):
        batched_sessions.append(sessions[i:i+batch_size])
    return batched_sessions

# Function to compute a hash ID from text
def compute_hash_id(text):
    # Use SHA-256 to generate a hash
    text = text + '_text'
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()  # Return hash as a hex string

def convert_attribute(value):
    """ Convert attributes to GDS-compatible types. """
    if isinstance(value, list):
        return [str(v) for v in value]
    elif isinstance(value, (int, float)):
        return value
    else:
        return str(value)

def clean_text(text):
    # remove NUL as well
    
    new_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\v", " ").replace("\f", " ").replace("\b", " ").replace("\a", " ").replace("\\e", " ").replace(";", ",")
    new_text = new_text.replace("\x00", "")
    new_text = re.sub(r'\s+', ' ', new_text).strip()

    return new_text

def remove_NUL(text):
    return text.replace("\x00", "")

def build_batched_events(all_node_list, batch_size):
    "The types are in Entity Event Relation"
    event_nodes = [node[0] for node in all_node_list if node[1].lower() == "event"]
    batched_events = []

    for i in range(0, len(event_nodes), batch_size):
        batched_events.append(event_nodes[i:i+batch_size])
    
    return batched_events

def build_batched_entities(all_node_list, batch_size):
    
    entity_nodes = [node[0] for node in all_node_list if node[1].lower() == "entity"]
    batched_entities = []

    for i in range(0, len(entity_nodes), batch_size):
        batched_entities.append(entity_nodes[i:i+batch_size])
    
    return batched_entities

def build_batched_relations(all_node_list, batch_size):

    relations = [node[0] for node in all_node_list if node[1].lower() == "relation"]
    # relations = list(set(relations))
    batched_relations = []

    for i in range(0, len(relations), batch_size):
        batched_relations.append(relations[i:i+batch_size])
    
    return batched_relations

def batched_inference(model:LLMGenerator, inputs, record=False, **kwargs):
    responses = model.generate_response(inputs, return_text_only = not record,  **kwargs)
    answers = []
    if record:
        text_responses = [response[0] for response in responses]
        usages = [response[1] for response in responses]
    else:
        text_responses = responses
    for i in range(len(text_responses)):
        answer = text_responses[i]
        answers.append([x.strip().lower() for x in answer.split(",")])
    
    if record:
        return answers, usages
    else:
        return answers

def load_data_with_shard(input_file, shard_idx, num_shards):

    with open(input_file, "r") as f:
        csv_reader = list(csv.reader(f))
    
    # data = csv_reader  
    data = csv_reader[1:]
    # Random shuffle the data before splitting into shards
    random.shuffle(data)
    
    total_lines = len(data)
    lines_per_shard = (total_lines + num_shards - 1) // num_shards
    start_idx = shard_idx * lines_per_shard
    end_idx = min((shard_idx + 1) * lines_per_shard, total_lines)
    
    return data[start_idx:end_idx]


CONCEPT_CSV_HEADER = ["node", "conceptualized_node", "node_type"]


def load_concept_checkpoint(output_file):
    """Return durable concept rows and repair an interrupted final CSV row.

    Concept output is flushed after every completed LLM batch. A Colab
    disconnect can therefore leave, at worst, an incomplete final CSV row.
    Earlier malformed rows indicate a genuinely corrupt checkpoint and are
    rejected instead of silently skipping work.
    """
    output_file = Path(output_file)
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return set()

    with output_file.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return set()
    if rows[0] != CONCEPT_CSV_HEADER:
        raise ValueError(f"Invalid concept checkpoint header: {output_file}")

    valid_rows = [CONCEPT_CSV_HEADER]
    completed = set()
    repaired = False
    for index, row in enumerate(rows[1:], 1):
        valid = (
            len(row) == 3
            and bool(row[0].strip())
            and row[2].strip().lower() in {"entity", "event", "relation"}
        )
        if not valid:
            if index != len(rows) - 1:
                raise ValueError(
                    f"Corrupt concept checkpoint row {index + 1}: {output_file}"
                )
            repaired = True
            continue
        key = (row[0], row[2].strip().lower())
        if key in completed:
            repaired = True
            continue
        completed.add(key)
        valid_rows.append(row)

    if repaired:
        temporary = output_file.with_suffix(output_file.suffix + ".repair.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(valid_rows)
        temporary.replace(output_file)
        print(f"Repaired trailing/duplicate concept checkpoint rows: {output_file}", flush=True)
    return completed


def write_concept_progress(output_folder, shard, total, completed, complete=False):
    target = Path(output_folder) / f"concept_progress_shard_{shard}.json"
    payload = {
        "shard": shard,
        "completed_nodes": completed,
        "total_nodes": total,
        "complete": complete,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)

def generate_concept(model: LLMGenerator,
            input_file = 'processed_data/triples_csv', 
            output_folder = 'processed_data/triples_conceptualized', 
            output_file = 'output.json', 
            logging_file = 'processed_data/logging.txt', 
            config:ProcessingConfig=None, 
            batch_size=32, 
            shard=0, 
            num_shards=1,
            **kwargs):
    log_dir = os.path.dirname(logging_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the log file if it doesn't exist
    if not os.path.exists(logging_file):
        open(logging_file, 'w').close()

    language = kwargs.get('language', 'en')
    record = kwargs.get('record', False)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    with open(f"{config.output_directory}/kg_graphml/{config.filename_pattern}_without_concept.pkl", "rb") as f: 
        temp_kg = pickle.load(f)

    # read data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    all_missing_nodes = load_data_with_shard(
        input_file,
        shard_idx=shard,
        num_shards=num_shards
    )

    output_file = Path(output_folder) / f"{output_file.rsplit('.', 1)[0]}_shard_{shard}.csv"
    completed_nodes = load_concept_checkpoint(output_file)
    input_keys = {(row[0], row[1].strip().lower()) for row in all_missing_nodes}
    stale_nodes = completed_nodes - input_keys
    if stale_nodes:
        raise ValueError(
            f"Concept checkpoint contains {len(stale_nodes)} nodes absent from current input; "
            "refusing to mix construction configurations"
        )
    pending_nodes = [
        row for row in all_missing_nodes
        if (row[0], row[1].strip().lower()) not in completed_nodes
    ]
    if completed_nodes:
        print(
            f"Resuming concept generation after {len(completed_nodes)}/{len(input_keys)} durable nodes.",
            flush=True,
        )
    write_concept_progress(
        output_folder, shard, len(input_keys), len(completed_nodes),
        complete=len(completed_nodes) == len(input_keys),
    )
        
    batched_events = build_batched_events(pending_nodes, batch_size)
    batched_entities = build_batched_entities(pending_nodes, batch_size)
    batched_relations = build_batched_relations(pending_nodes, batch_size)
    
    all_batches = []
    all_batches.extend(('event', batch) for batch in batched_events)
    all_batches.extend(('entity', batch) for batch in batched_entities)
    all_batches.extend(('relation', batch) for batch in batched_relations)
    
    print("all_batches", len(all_batches))



    
    new_file = not output_file.exists() or output_file.stat().st_size == 0
    with output_file.open("a", newline='', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        if new_file:
            csv_writer.writerow(CONCEPT_CSV_HEADER)
            file.flush()

        # for batch_type, batch in tqdm(all_batches, total=total_batches, desc="Generating concepts"):
        # don't use tqdm for now
        for batch_type, batch in tqdm(all_batches, desc="Shard_{}".format(shard)):
            # print("batch_type", batch_type)
            # print("batch", batch)
            replace_context_token = None
            if batch_type == 'event':
                template = CONCEPT_INSTRUCTIONS[language]['event']
                node_type = 'event'
                replace_token = '[EVENT]'
            elif batch_type == 'entity':
                template = CONCEPT_INSTRUCTIONS[language]['entity']
                node_type = 'entity'
                replace_token = '[ENTITY]'
                replace_context_token = '[CONTEXT]'
            elif batch_type == 'relation':
                template = CONCEPT_INSTRUCTIONS[language]['relation']
                node_type = 'relation'
                replace_token = '[RELATION]'

            inputs = []
            for node in batch:
                # sample node from given node and replace context token.
                if replace_context_token:
                    node_id = get_node_id(node)
                    entity_predecessors = list(temp_kg.predecessors(node_id))
                    entity_successors = list(temp_kg.successors(node_id))

                    context = ""

                    if len(entity_predecessors) > 0:
                        random_two_neighbors = random.sample(entity_predecessors, min(1, len(entity_predecessors)))
                        context += ", ".join([f"{temp_kg.nodes[neighbor]['id']} {temp_kg[neighbor][node_id]['relation']}" for neighbor in random_two_neighbors])
                    
                    if len(entity_successors) > 0:
                        random_two_neighbors = random.sample(entity_successors, min(1, len(entity_successors)))
                        context += ", ".join([f"{temp_kg[node_id][neighbor]['relation']} {temp_kg.nodes[neighbor]['id']}" for neighbor in random_two_neighbors])
                    
                    prompt = template.replace(replace_token, node).replace(replace_context_token, context)
                else:
                    prompt = template.replace(replace_token, node)
                constructed_input = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": f"{prompt}"},
                ]
                inputs.append(constructed_input)

            try:
                # print("inputs", inputs)
                if record:
                    # If recording, we will get both answers and responses
                    answers, usages = batched_inference(model, inputs, record=record, max_workers = config.max_workers)
                else:
                    answers = batched_inference(model, inputs, record=record, max_workers = config.max_workers)
                    usages = None
                # print("answers", answers)
            except Exception as e:
                logging.error(f"Error processing {batch_type} batch: {e}")
                raise e
            # try:
            #     answers = batched_inference(llm, sampling_params, inputs)
            # except Exception as e:
            #     logging.error(f"Error processing {batch_type} batch: {e}")
            #     continue

            for i,(node, answer) in enumerate(zip(batch, answers)):
                # print(node, answer, node_type)
                if usages is not None:
                    logging.info(f"Usage log: Node {node}, completion_usage: {usages[i]}")
                csv_writer.writerow([node, ", ".join(answer), node_type])
                file.flush()
                completed_nodes.add((node, node_type))
            write_concept_progress(
                output_folder, shard, len(input_keys), len(completed_nodes),
                complete=len(completed_nodes) == len(input_keys),
            )
    if len(completed_nodes) != len(input_keys):
        raise RuntimeError(
            f"Concept generation returned incomplete output: "
            f"{len(completed_nodes)}/{len(input_keys)} nodes"
        )
    write_concept_progress(
        output_folder, shard, len(input_keys), len(completed_nodes), complete=True
    )
    # count unique conceptualized nodes
    conceptualized_nodes = []

    conceptualized_events = []
    conceptualized_entities = []
    conceptualized_relations = []

    with output_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            conceptualized_nodes.extend(row[1].split(","))
            if row[2] == "event":
                conceptualized_events.extend(row[1].split(","))
            elif row[2] == "entity":
                conceptualized_entities.extend(row[1].split(","))
            elif row[2] == "relation":
                conceptualized_relations.extend(row[1].split(","))
    
    conceptualized_nodes = [x.strip() for x in conceptualized_nodes]
    conceptualized_events = [x.strip() for x in conceptualized_events]
    conceptualized_entities = [x.strip() for x in conceptualized_entities]
    conceptualized_relations = [x.strip() for x in conceptualized_relations]

    unique_conceptualized_nodes = list(set(conceptualized_nodes))
    unique_conceptualized_events = list(set(conceptualized_events))
    unique_conceptualized_entities = list(set(conceptualized_entities))
    unique_conceptualized_relations = list(set(conceptualized_relations))

    print(f"Number of unique conceptualized nodes: {len(unique_conceptualized_nodes)}")
    print(f"Number of unique conceptualized events: {len(unique_conceptualized_events)}")
    print(f"Number of unique conceptualized entities: {len(unique_conceptualized_entities)}")
    print(f"Number of unique conceptualized relations: {len(unique_conceptualized_relations)}")

    return 
    


