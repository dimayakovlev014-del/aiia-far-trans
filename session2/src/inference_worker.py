"""Persistent isolated Metal model worker; stdout is JSON, native diagnostics stderr."""
import json
import hashlib
from pathlib import Path
import resource
import sys
import time
import traceback

from prepare_data import ROOT, sha


def emit(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    config=json.loads((ROOT/'session2/configs/decoding.json').read_text())
    model=json.loads((ROOT/'session2/configs/model.json').read_text())
    path=ROOT/'session2/models'/model['filename']
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError('Requested Metal backend is unavailable; no silent CPU fallback')
    if sha(path)!=model['sha256']:
        raise ValueError('Model hash mismatch')
    start=time.perf_counter()
    llm=llama_cpp.Llama(model_path=str(path),verbose=True,
                        **{k:config[k] for k in ('n_ctx','n_gpu_layers','n_threads','n_batch','n_ubatch','flash_attn','seed')})
    template=llm.metadata['tokenizer.chat_template']
    formatter=Jinja2ChatFormatter(template=template,eos_token=llm.detokenize([llm.token_eos()],special=True).decode(),
                                bos_token='',add_generation_prompt=True)
    version=llama_cpp.__version__
    lib=Path(llama_cpp.__file__).parent/'lib'
    backend=dict(llama_cpp_python=version,system_info=llama_cpp.llama_print_system_info().decode(),
                 model_context_train=llm.metadata.get('qwen2.context_length'),
                 architecture=llm.metadata.get('general.architecture'),
                 gpu_offload_supported=llama_cpp.llama_supports_gpu_offload(),
                 template=template,template_sha256=__import__('hashlib').sha256(template.encode()).hexdigest(),
                 libraries={p.name:sha(p) for p in lib.glob('*.dylib')},
                 load_seconds=time.perf_counter()-start)
    emit(dict(status='ready',backend=backend))
    def tokenize(prompt):
        text=formatter(messages=[dict(role='user',content=prompt)]).prompt
        return llm.tokenize(text.encode(),add_bos=False,special=True)
    for line in sys.stdin:
        request=json.loads(line)
        try:
            if request.get('action')=='count':
                emit(dict(status='counted',tokens=[len(tokenize(p)) for p in request['prompts']]))
                continue
            prompt=request['prompt']
            tokens=tokenize(prompt)
            if len(tokens)+config['max_tokens']>config['n_ctx']:
                emit(dict(status='context_overflow',input_tokens=len(tokens),error='Complete prompt does not fit; no truncation'))
                continue
            llm.reset()
            start=time.perf_counter()
            response=llm.create_completion(prompt=tokens,
                **{k:config[k] for k in ('max_tokens','temperature','top_p','top_k','min_p','repeat_penalty','seed','stop')})
            emit(dict(status='generated',text=response['choices'][0]['text'],
                      user_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      input_token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                      finish_reason=response['choices'][0]['finish_reason'],
                      input_tokens=len(tokens),output_tokens=response['usage']['completion_tokens'],
                      inference_seconds=time.perf_counter()-start,
                      peak_process_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        except Exception as exc:
            emit(dict(status='generation_exception',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        emit(dict(status='model_load_error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise
