import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def answer_question_with_chunk(user_question, chunk_text, model_path):
    """
    Answer user question based on provided chunk using GPT-2 with time tracking
    """
    # Start total time tracking
    total_start_time = time.time()
    times = {}

    try:
        # Model loading time
        load_start = time.time()
        print("Loading GPT-2 model...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_time = time.time() - load_start
        times['model_loading'] = load_time
        print(f"Model loaded in {load_time:.2f} seconds")

        # Prompt preparation time
        prompt_start = time.time()
        prompt = f"""Context: {chunk_text}

Question: {user_question}

Answer based on the context:"""

        print(f"User Question: {user_question}")
        prompt_time = time.time() - prompt_start
        times['prompt_preparation'] = prompt_time

        # Tokenization time
        tokenize_start = time.time()
        inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        tokenize_time = time.time() - tokenize_start
        times['tokenization'] = tokenize_time

        print(f"Input tokens: {inputs.shape[1]}")
        print("\nGenerating answer...")

        # Text generation time
        generation_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        generation_time = time.time() - generation_start
        times['text_generation'] = generation_time

        # Decoding time
        decode_start = time.time()
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decode_time = time.time() - decode_start
        times['decoding'] = decode_time

        # Post-processing time
        postprocess_start = time.time()
        answer_start = full_response.find("Answer based on the context:") + len("Answer based on the context:")
        answer = full_response[answer_start:].strip()
        postprocess_time = time.time() - postprocess_start
        times['post_processing'] = postprocess_time

        # Calculate total time
        total_time = time.time() - total_start_time
        times['total_time'] = total_time

        # Print timing breakdown
        print(f"\n{'=' * 50}")
        print("TIMING BREAKDOWN:")
        print(f"{'=' * 50}")
        print(f"Model Loading:     {times['model_loading']:.3f}s")
        print(f"Prompt Prep:       {times['prompt_preparation']:.3f}s")
        print(f"Tokenization:      {times['tokenization']:.3f}s")
        print(f"Text Generation:   {times['text_generation']:.3f}s")
        print(f"Decoding:          {times['decoding']:.3f}s")
        print(f"Post-processing:   {times['post_processing']:.3f}s")
        print(f"{'=' * 50}")
        print(f"TOTAL TIME:        {times['total_time']:.3f}s")
        print(f"{'=' * 50}")

        # Calculate tokens per second
        output_tokens = outputs.shape[1] - inputs.shape[1]
        if generation_time > 0:
            tokens_per_second = output_tokens / generation_time
            print(f"Generated tokens:  {output_tokens}")
            print(f"Generation speed:  {tokens_per_second:.2f} tokens/second")
            print(f"{'=' * 50}")

        return answer, times

    except Exception as e:
        total_time = time.time() - total_start_time
        print(f"Error after {total_time:.2f} seconds: {e}")
        return f"Error generating answer: {e}", {'total_time': total_time, 'error': True}


# Test with your example
if __name__ == "__main__":
    model_path = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\plugins\ai_modules\GPT-2"

    user_question = "what can I remove with trichloroacetic acid"

    chunk_text = """used this method, in combination with enzymatic tech- niques,to synthesize a 126-nucleotide tRNA gene,a project that required several years of intense effort by numerous skilled chemists. a. The Phosphoramidite Method By the early 1980s, these difficult and time-consuming processes had been supplanted by much faster solid phase methodologies that permitted oligonucleotide synthesis to be automated. The presently most widely used chemistry, which was formulated by Robert Letsinger and further de- veloped by Marvin Caruthers, is known as the phosphor- amidite method. This series of nonaqueous reactions adds a single nucleotide to a growing oligonucleotide chain as follows (Fig. 7-38): 1. The dimethoxytrityl (DMTr) protecting group at the 5¿ end of the growing oligonucleotide chain (which is anchored via a linking group at its 3¿ end to a solid support, S) is removed by treatment with an acid such as trichloroacetic acid (Cl3CCOOH). 2. The newly liberated 5¿ end of the"""

    answer, timing_data = answer_question_with_chunk(user_question, chunk_text, model_path)

    print(f"\nFINAL ANSWER: {answer}")