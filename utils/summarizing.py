import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import math
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def summarize(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    word_count = len(text.split())
    min_length = int(10 * math.sqrt(word_count + 64) - 80)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=700, min_length=min_length, length_penalty=1.7, num_beams=4)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

text = """Shrek is an anti-social ogre who loves the solitude of his swamp and enjoys fending off mobs and intruders. One day, his life is interrupted after he inadvertently saves a talkative Donkey from some soldiers, prompting Donkey to forcibly stay with him. Donkey is one of many fairytale creatures that are being exiled or sold by the dwarfish Lord Farquaad of Duloc to beautify his land. However, the creatures inadvertently end up in the swamp. Angered by the intrusion, Shrek resolves to visit Farquaad and demand that he moves the creatures elsewhere, reluctantly allowing Donkey to accompany him as he is the only one who knows where Duloc is.
    Meanwhile, Farquaad is presented with the Magic Mirror, who tells him that he must marry a princess in order to become king. Farquaad randomly chooses Princess Fiona, who is imprisoned in a castle guarded by a Dragon. Unwilling to rescue Fiona himself, he organizes a tournament in which the winner will receive the "privilege" of performing the task on his behalf. When Shrek and Donkey arrive at Duloc, Farquaad announces that whoever kills Shrek will win the tournament; however, Shrek and Donkey defeat Farquaad's knights with relative ease. Amused, Farquaad proclaims them champions, and agrees to relocate the fairytale creatures if Shrek rescues Fiona.
    Shrek and Donkey travel to the castle and the Dragon attacks them. Shrek locates Fiona, who is appalled by his lack of romanticism; they flee the castle after rescuing Donkey from the Dragon, who is female and becomes smitten with him after he flatters her. When she discovers Shrek is an ogre, Fiona stubbornly refuses to go to Duloc, demanding Farquaad arrive in person to save her; Shrek carries Fiona against her will. That night, after setting up camp, and with Fiona alone in a cave, Shrek admits to Donkey that he is anti-social because he grew frustrated after being constantly judged for his appearance. Fiona overhears this and becomes kinder to Shrek. The next day, Robin Hood and the Merry Men try to "rescue" Fiona from Shrek, but she easily defeats them in physical combat. Shrek becomes impressed with Fiona, and they begin to fall in love.
    When the trio nears Duloc, Fiona takes shelter in a windmill for the evening. Donkey enters alone and discovers that Fiona has transformed into an ogre. She explains that during her childhood, she was cursed to transform into an ogre at night but retain her human form during the day. She tells Donkey that only "true love's kiss" will break the spell and change her to "love's true form". Meanwhile, Shrek is about to confess his feelings to Fiona, when he overhears Fiona referring to herself as an "ugly beast". Believing that she is talking about him, Shrek angrily leaves and returns the next morning with Farquaad. Confused and hurt by Shrek's abrupt hostility, Fiona reluctantly accepts Farquaad's marriage proposal and requests that they be married that day before sunset. Shrek angrily dismisses Donkey and returns to his now vacated swamp but quickly realizes that he feels miserable without Fiona. A frustrated Donkey scolds Shrek for jumping to conclusions and reveals that Fiona was not referring to him as an "ugly beast", although Donkey does not reveal Fiona's secret to Shrek. The two reconcile, and Donkey summons the Dragon, whom he had reunited with earlier in the day. Shrek and Donkey ride Dragon to Duloc so they can stop the wedding.
    Shrek interrupts the ceremony just before it ends. Before they can kiss, the sun sets, and Fiona transforms into an ogre in front of everyone. Disgusted and enraged, Farquaad orders Shrek to be executed and Fiona re-imprisoned, so that he will still be king by technicality. The two are saved when the Dragon, ridden by Donkey, breaks in and devours Farquaad. After declaring their love, Shrek and Fiona kiss and the curse is broken, but she remains an ogre due to her true love being Shrek; Shrek reassures her that she is beautiful in her ogre form. They marry in the swamp with the fairytale creatures in attendance, while Donkey returns Dragon's feelings, then leave for their honeymoon."""
