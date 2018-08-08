import matplotlib.pyplot as plt
in_pic = []
with open('input_origin.txt','r') as fin:
    lines = fin.readlines()

#画图
pic_uins = []
temp = []
cnt = 0
for line in lines:
    a = int(line.rstrip(),16)
    temp.append(a)
    cnt+=1
    if cnt == 32:
        pic_uins.append(temp)
        temp=[]
        cnt=0
with open('input_standard.txt','w') as fout:
    for ver in range(32):
        fout.write("%02d:" % ver)
        for hor in range(32):
            fout.write(" %04x" % pic_uins[ver][hor])
        fout.write("\n")

plt.subplot(121)
plt.imshow(pic_uins,cmap='gray')
plt.axis('off')

pic = []
temp = []
cnt = 0
cntl = 0
for line in lines:
    a = int(line.rstrip(),16)
    temp.append(a)
    cnt+=1
    if cnt == 32:
        pic.append(temp)
        temp=[]
        cnt=0
        cntl +=1
        if cntl % 4 == 0:
            pic.append([0x400]*39)
    elif cnt % 4 == 0:
        temp.append(0x400)
plt.subplot(122)
plt.imshow(pic,cmap='gray')
plt.axis('off')
plt.show()

addr=0
with open('input.txt','w') as fout:
    for vpic in range(8):
        for hpic in range(8):
            fout.write("@%08x" % addr)
            for v in range(4):
                for h in range(4):
                    index = (vpic*4 + v)*32 + hpic*4 + h
                    num = lines[index].rstrip()
                    fout.write(" " + num)
            fout.write("\n")
            addr = addr + 16
