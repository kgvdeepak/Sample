from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import base64
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
import math
import uvicorn

class AudioRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+j2xm0hByaFyvLCdSYGII3N8dWREAobZDXn3aRSFhB"
            }
        }
    )

    audio: str

class AudioResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "processed_audio": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+j2xm0hByaFyvLCdSYGII3N8dWREAobZDXn3aRSFhB"
            }
        }
    )

    processed_audio: str

app = FastAPI(
    title="High-Quality Silence Removal API",
    description="An API that removes silence from audio files with superior speech preservation. Optimized for conversational audio with 5-minute timeout protection. Uses direct whole-file processing for files under 5 minutes for maximum quality, and chunked processing for longer files. Removes silence periods longer than 5 seconds while preserving natural speech pauses.",
    version="2.3.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Global process pool for parallel processing - optimized workers based on CPU cores
import os
# Reduced worker count for better stability and memory management
process_pool = ProcessPoolExecutor(max_workers=min(os.cpu_count() // 2 or 1, 2))

def process_audio_chunk(chunk_data: bytes, chunk_index: int, chunk_start_ms: int, sample_rate: int, sample_width: int) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Process a single audio chunk to detect non-silent segments.
    Returns (chunk_index, segments) where segments are (start_ms, end_ms) relative to original audio.
    Uses ultra-conservative silence detection to preserve speech details and articulation.
    """
    try:
        chunk_buffer = BytesIO(chunk_data)
        audio_chunk = AudioSegment.from_raw(chunk_buffer, sample_width=sample_width, frame_rate=sample_rate, channels=1)
        
        # Convert to numpy array
        samples = np.array(audio_chunk.get_array_of_samples(), dtype=np.float64)
        
        if len(samples) == 0:
            return (chunk_index, [])
        
        # Calculate silence threshold using dBFS approach (more reliable)
        audio_dbfs = audio_chunk.dBFS
        # Ultra-conservative silence detection to preserve all speech content
        silence_threshold_db = -55.0  # Start with extremely conservative threshold
        if audio_dbfs < -35:
            silence_threshold_db = audio_dbfs + 35  # very quiet audio
        elif audio_dbfs > -15:
            silence_threshold_db = -45.0
        else:
            silence_threshold_db = -47.0
        
        # Convert dB threshold to amplitude
        max_possible = float(1 << (8 * sample_width - 1))
        threshold_amp = max_possible * (10 ** (silence_threshold_db / 20))
        
        print(f"Chunk {chunk_index}: Audio dBFS={audio_dbfs:.1f}, Silence threshold={silence_threshold_db:.1f}dB, Threshold amp={threshold_amp:.2f}")
        
        # Find samples above threshold (non-silent)
        abs_samples = np.abs(samples)
        loud = abs_samples > threshold_amp
        
        if not np.any(loud):
            print(f"Chunk {chunk_index}: No audio above threshold - this chunk is silent")
            return (chunk_index, [])
        
        # Apply optimized smoothing
        window_size = int(sample_rate * 0.05)  # 50ms window
        if window_size > 1 and len(loud) > window_size:
            if len(loud) < 50000:
                loud_smoothed = np.convolve(loud.astype(float), np.ones(window_size)/window_size, mode='same')
                loud = loud_smoothed > 0.1
            else:
                batch_size = 1000000
                for i in range(0, len(loud), batch_size):
                    end_idx = min(i + batch_size, len(loud))
                    if end_idx - i > window_size:
                        batch = loud[i:end_idx].astype(float)
                        cumsum = np.cumsum(np.insert(batch, 0, 0))
                        avg_window = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                        loud[i:i+len(avg_window)] = avg_window > 0.1
        
        # Find contiguous loud regions
        loud_diff = np.diff(np.concatenate(([False], loud, [False])).astype(int))
        starts = np.where(loud_diff == 1)[0]
        ends = np.where(loud_diff == -1)[0]
        
        segments = []
        for start_idx, end_idx in zip(starts, ends):
            start_ms = chunk_start_ms + int((start_idx / sample_rate) * 1000)
            end_ms = chunk_start_ms + int((end_idx / sample_rate) * 1000)
            if end_ms - start_ms > 25:
                segments.append((start_ms, end_ms))
                print(f"Chunk {chunk_index}: Found segment {start_ms}ms - {end_ms}ms (duration: {end_ms - start_ms}ms)")
        
        print(f"Chunk {chunk_index}: Found {len(segments)} valid segments")
        return (chunk_index, segments)
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {e}")
        return (chunk_index, [])

def merge_segments_remove_long_silence(segments: List[Tuple[int, int]], min_silence_duration_ms: int = 5000) -> List[Tuple[int, int]]:
    """Merge segments and remove silence periods longer than min_silence_duration_ms."""
    if not segments:
        return []
    segments = sorted(segments)
    merged_segments = []
    current_start, current_end = segments[0]
    for start, end in segments[1:]:
        gap = start - current_end
        if gap < min_silence_duration_ms:
            current_end = max(current_end, end)
        else:
            extended_end = current_end + min(gap // 2, 1200)
            merged_segments.append((current_start, extended_end))
            extended_start = max(current_end, start - min(gap // 2, 1200))
            current_start, current_end = extended_start, end
    merged_segments.append((current_start, current_end))
    final_segments = []
    for start, end in merged_segments:
        if end - start > 50:
            final_segments.append((start, end))
    return final_segments

async def process_audio_chunks_async(audio_segment: AudioSegment, chunk_duration_ms: int = 10000) -> List[Tuple[int, int]]:
    """Process audio in chunks asynchronously."""
    audio_duration_ms = len(audio_segment)
    sample_rate = audio_segment.frame_rate
    sample_width = audio_segment.sample_width
    print(f"Starting chunk processing: Audio duration={audio_duration_ms}ms, Chunk size={chunk_duration_ms}ms")
    mono_audio = audio_segment.set_channels(1)
    chunk_tasks = []
    loop = asyncio.get_event_loop()
    total_chunks = math.ceil(audio_duration_ms / chunk_duration_ms)
    print(f"Total chunks to process: {total_chunks}")
    for i in range(0, audio_duration_ms, chunk_duration_ms):
        chunk_start_ms = i
        chunk_end_ms = min(i + chunk_duration_ms, audio_duration_ms)
        chunk_segment = mono_audio[chunk_start_ms:chunk_end_ms]
        chunk_data = chunk_segment.raw_data
        task = loop.run_in_executor(
            process_pool,
            process_audio_chunk,
            chunk_data,
            i // chunk_duration_ms,
            chunk_start_ms,
            sample_rate,
            sample_width
        )
        chunk_tasks.append(task)
    try:
        print("Processing chunks...")
        chunk_results = await asyncio.wait_for(
            asyncio.gather(*chunk_tasks, return_exceptions=True),
            timeout=300.0
        )
        print("Chunk processing completed!")
    except asyncio.TimeoutError:
        print("Chunk processing timed out after 5 minutes")
        raise HTTPException(status_code=408, detail="Audio processing timed out. File may be too large or complex.")
    all_segments = []
    for result in chunk_results:
        if isinstance(result, Exception):
            print(f"Chunk processing error: {result}")
            continue
        chunk_index, segments = result
        all_segments.extend(segments)
    return all_segments

async def process_audio_direct(audio_segment: AudioSegment) -> List[Tuple[int, int]]:
    """Process audio directly without chunking."""
    print("Processing audio as a whole file for maximum quality...")
    sample_rate = audio_segment.frame_rate
    sample_width = audio_segment.sample_width
    mono_audio = audio_segment.set_channels(1)
    audio_dbfs = mono_audio.dBFS
    if audio_dbfs < -35:
        silence_threshold_db = audio_dbfs + 35
    elif audio_dbfs > -15:
        silence_threshold_db = -45.0
    else:
        silence_threshold_db = -48.0
    print(f"Whole file: Audio dBFS={audio_dbfs:.1f}, Using silence threshold={silence_threshold_db:.1f}dB")
    samples = np.array(mono_audio.get_array_of_samples(), dtype=np.float64)
    max_possible = float(1 << (8 * sample_width - 1))
    threshold_amp = max_possible * (10 ** (silence_threshold_db / 20))
    abs_samples = np.abs(samples)
    loud = abs_samples > threshold_amp
    if not np.any(loud):
        print("No audio above threshold - entire file is silent")
        return []
    window_size = int(sample_rate * 0.05)
    if window_size > 1 and len(loud) > window_size:
        batch_size = 2000000
        for i in range(0, len(loud), batch_size):
            end_idx = min(i + batch_size, len(loud))
            if end_idx - i > window_size:
                batch = loud[i:end_idx].astype(float)
                cumsum = np.cumsum(np.insert(batch, 0, 0))
                avg_window = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                loud[i:i+len(avg_window)] = avg_window > 0.1
    loud_diff = np.diff(np.concatenate(([False], loud, [False])).astype(int))
    starts = np.where(loud_diff == 1)[0]
    ends = np.where(loud_diff == -1)[0]
    segments = []
    for start_idx, end_idx in zip(starts, ends):
        start_ms = int((start_idx / sample_rate) * 1000)
        end_ms = int((end_idx / sample_rate) * 1000)
        if end_ms - start_ms > 15:
            segments.append((start_ms, end_ms))
            print(f"Whole file: Found segment {start_ms}ms - {end_ms}ms (duration: {end_ms - start_ms}ms)")
    print(f"Whole file: Found {len(segments)} valid segments")
    return segments

@app.get("/", summary="Root endpoint", description="Welcome message and API information")
async def root():
    return {
        "message": "Welcome to the Fast Chunked Silence Removal API",
        "version": "2.2.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.post("/process_audio_vectorized",
          response_model=AudioResponse,
          summary="Remove silence from audio",
          description="Processes an M4A audio file to remove silent parts using async chunked processing. Optimized for conversational audio - removes silence periods longer than 5 seconds while preserving natural speech pauses and quiet speech. Processes audio in 20-second chunks for optimal performance.",
          responses={
              200: {
                  "description": "Successfully processed audio",
                  "content": {
                      "application/json": {
                          "example": {
                              "processed_audio": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+j2xm0hByaFyvLCdSYGII3N8dWREAobZDXn3aRSFhB"
                          }
                      }
                  }
              },
              400: {
                  "description": "Bad request - invalid audio data or no non-silent audio found",
                  "content": {
                      "application/json": {
                          "example": {
                              "detail": "Invalid base64 audio data: Invalid base64-encoded string: ..."
                          }
                      }
                  }
              }
          })
async def process_audio_vectorized(request: AudioRequest):
    try:
        audio_data = base64.b64decode(request.audio)
        print(f"Received audio data size: {len(audio_data)} bytes ({len(audio_data)/(1024*1024):.1f} MB)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {e}")
    if len(audio_data) > 50 * 1024 * 1024:
        print("WARNING: Large audio file detected. Processing may take several minutes.")
        raise HTTPException(status_code=413, detail="Audio file too large. Please use files smaller than 50MB for optimal performance.")
    audio_file = BytesIO(audio_data)
    try:
        audio_segment = AudioSegment.from_file(audio_file, format="m4a")
        print(f"Audio loaded: Duration={len(audio_segment)}ms, Sample rate={audio_segment.frame_rate}Hz, Channels={audio_segment.channels}")
        duration_minutes = len(audio_segment) / (1000 * 60)
        if duration_minutes > 30:
            print(f"WARNING: Long audio file ({duration_minutes:.1f} minutes) - processing may take time")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {e}")
    all_segments = []
    audio_duration_ms = len(audio_segment)
    if audio_duration_ms <= 5 * 60 * 1000:
        print("Using direct whole-file processing for better quality...")
        all_segments = await process_audio_direct(audio_segment)
    else:
        print("Using chunked processing for long file...")
        all_segments = await process_audio_chunks_async(audio_segment)
    if not all_segments:
        print("No segments found with chunked processing, trying fallback method...")
        mono_audio = audio_segment.set_channels(1)
        audio_dbfs = mono_audio.dBFS
        if audio_dbfs < -35:
            silence_threshold_db = audio_dbfs + 30
        elif audio_dbfs > -15:
            silence_threshold_db = -38.0
        else:
            silence_threshold_db = -42.0
        print(f"Fallback: Audio dBFS={audio_dbfs:.1f}, Using silence threshold={silence_threshold_db:.1f}dB")
        samples = np.array(mono_audio.get_array_of_samples(), dtype=np.float64)
        max_possible = float(1 << (8 * mono_audio.sample_width - 1))
        threshold_amp = max_possible * (10 ** (silence_threshold_db / 20))
        loud = np.abs(samples) > threshold_amp
        if np.any(loud):
            window_size = int(mono_audio.frame_rate * 0.05)
            if window_size > 1:
                batch_size = 1000000
                for i in range(0, len(loud), batch_size):
                    end_idx = min(i + batch_size, len(loud))
                    batch = loud[i:end_idx].astype(float)
                    cumsum = np.cumsum(np.insert(batch, 0, 0))
                    avg_window = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                    loud[i:end_idx-window_size+1] = avg_window > 0.1
            loud_diff = np.diff(np.concatenate(([False], loud, [False])).astype(int))
            starts = np.where(loud_diff == 1)[0]
            ends = np.where(loud_diff == -1)[0]
            for start_idx, end_idx in zip(starts, ends):
                start_ms = int((start_idx / mono_audio.frame_rate) * 1000)
                end_ms = int((end_idx / mono_audio.frame_rate) * 1000)
                if end_ms - start_ms > 50:
                    all_segments.append((start_ms, end_ms))
                    print(f"Fallback found segment: {start_ms}ms - {end_ms}ms (duration: {end_ms - start_ms}ms)")
    if not all_segments:
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)
        max_amplitude = np.max(np.abs(samples))
        rms = np.sqrt(np.mean(samples**2))
        print(f"Audio analysis - Max amplitude: {max_amplitude:.2f}, RMS: {rms:.2f}")
        if max_amplitude < 1e-3:
            raise HTTPException(status_code=400, detail="The uploaded audio is completely silent.")
        elif rms < 1e-4:
            raise HTTPException(status_code=400, detail="The uploaded audio has extremely low signal levels.")
        else:
            raise HTTPException(status_code=400, detail="No non-silent audio segments found. The audio may have unusual characteristics or very low signal levels.")
    merged_segments = merge_segments_remove_long_silence(all_segments, min_silence_duration_ms=5000)
    print(f"Original segments: {len(all_segments)}")
    print(f"Merged segments: {len(merged_segments)}")
    for i, (start, end) in enumerate(merged_segments):
        print(f"Final segment {i+1}: {start}ms - {end}ms (duration: {end-start}ms)")
    if not merged_segments:
        raise HTTPException(status_code=400, detail="No valid audio segments found after processing.")
    original_duration = len(audio_segment)
    processed_duration = sum(end - start for start, end in merged_segments)
    silence_removed = original_duration - processed_duration
    print(f"Original duration: {original_duration}ms")
    print(f"Processed duration: {processed_duration}ms")
    print(f"Silence removed: {silence_removed}ms ({silence_removed/original_duration*100:.1f}%)")
    padding_ms = 2000
    audio_duration_ms = len(audio_segment)
    padded_segments = []
    for start, end in merged_segments:
        padded_start = max(0, start - padding_ms)
        padded_end = min(audio_duration_ms, end + padding_ms)
        padded_segments.append((padded_start, padded_end))
    if padded_segments:
        padded_segments = sorted(padded_segments)
        final_padded_segments = []
        current_start, current_end = padded_segments[0]
        for start, end in padded_segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                final_padded_segments.append((current_start, current_end))
                current_start, current_end = start, end
        final_padded_segments.append((current_start, current_end))
        padded_segments = final_padded_segments
    processed_audio = AudioSegment.empty()
    for start, end in padded_segments:
        processed_audio += audio_segment[start:end]
    output_buffer = BytesIO()
    processed_audio.export(output_buffer, format="mp4", codec="aac")
    output_bytes = output_buffer.getvalue()
    processed_audio_b64 = base64.b64encode(output_bytes).decode("utf-8")
    return {"processed_audio": processed_audio_b64}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
