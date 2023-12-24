import os

path = '/media/ihdav/Files/Research/Infrrd_Research/chatGPT/Data/Whatsapp_Chats'
chat_files = os.listdir(path)

text_out = ''

for file in chat_files:
    if file.endswith('.txt'):
        with open(os.path.join(path, file), 'r') as f:
            text_lines = f.readlines()

        new_lines = []
        i = 0

        while i<len(text_lines):
            temp_line = text_lines[i]
            
            for j in range(i+1,len(text_lines)):
                if text_lines[j][0:2].isdigit() and text_lines[j][2]=='/' and text_lines[j][5]=='/':
                    break
                else:
                    temp_line += '\n'+text_lines[j]
            
            i = j

            if (i==len(text_lines)-1):
                new_lines.append(temp_line)
                break

            new_lines.append(temp_line)
        
        processed_lines = []
        nuprocessed_lines = []

        for line in text_lines:
            if " - " in line:
                if (line.split(" - ")[1]!=''):
                    processed_lines.append(line.split(" - ")[1])
            else:
                nuprocessed_lines.append(line)
        
        
        for line in processed_lines:
            if (len(line.split(":",1))>1):
                if (line.split(":",1)[1]!=''):
                    text_out += line.split(":",1)[1]+'\n'

print(len(text_out.split('\n')))
with open("out_data_full.txt", "w", encoding="utf-8") as f:
    f.write(text_out)