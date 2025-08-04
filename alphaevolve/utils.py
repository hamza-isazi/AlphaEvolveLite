import multiprocessing
import functools
import traceback
import queue as py_queue

def timeout(seconds, error_message="Function call timed out"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target(queue, *a, **kw):
                try:
                    result = func(*a, **kw)
                    queue.put((True, result))
                except Exception as e:
                    # Format the exception with a traceback so we can see where the error occurred
                    tb = traceback.extract_tb(e.__traceback__)
                    if tb:
                        # Skip the first frame (this function) and include all remaining frames
                        frames = tb[1:]
                        error_message = str(e)
                        for frame in frames:
                            error_message += f"\n  File \"{frame.filename}\", line {frame.lineno}, in {frame.name}"
                            if frame.line:
                                error_message += f"\n    {frame.line.strip()}"
                    else:
                        error_message = str(e)
                    # Put the formatted error message in the queue as a tuple (False, exception)
                    queue.put((False, Exception(error_message)))

            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=target, args=(queue, *args), kwargs=kwargs)
            process.start()
            process.join(seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError(error_message)

            try:
                # If the subprocess is terminated, it will not return a result and block forever, so we need to raise a TimeoutError
                success, payload = queue.get(timeout=1)
            except py_queue.Empty:
                raise TimeoutError("Subprocess was terminated but did not return a result.")
            if success:
                return payload
            else:
                raise payload

        return wrapper
    return decorator 