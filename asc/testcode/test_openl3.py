import openl3
import soundfile as sf
# import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":

    audio, sr = sf.read('/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development/audio/airport-barcelona-0-3-a.wav')
    print(sr)
    print(audio)

    model = openl3.models.load_audio_embedding_model(content_type="music", input_repr="mel256", embedding_size=512)

    emb, ts = openl3.get_audio_embedding(audio, sr, model=model, hop_size=0.1)
    print(emb, ts)

    #openl3-music-mel256-emb512-hop0_1
