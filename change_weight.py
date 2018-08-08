in_pic = []
groups = [   ['Layer1_L1',5,1,6],
            ['Layer3_L3',5,6,16],
            ['Layer5_F1',400,120],
            ['Layer6_F2',120,84],
            ['Layer7_F3',84,10]
        ]
#width, height, chennel
reshape = (5,5,16)

with open('weight.txt','w') as fout:
    addr  = 0
    for group in groups:
        name = group[0]
        print("%s: %x" % (name, int(addr/16)))
        #读数据
        with open (name + "_weight.txt") as f:
            weights = f.readlines()
        with open (name + "_bias.txt") as f:
            bias = f.readlines()
        if "_L" in name:
            #L1 or L2
            cnt=group[1]*group[1]*group[2]
            index = 0
            for box in range(group[3]):
                for i in range(cnt):
                    if (addr % 16) == 0:
                        fout.write("\n@%08x" % addr)
                    num = weights[index].rstrip()
                    fout.write(" " + num)
                    addr +=1
                    index = index + 1
                # 加index
                if (addr % 16) == 0:
                    fout.write("\n@%08x" % addr)
                num = bias[box].rstrip()
                fout.write(" " + num)
                addr +=1
                # box后面补0
                while (addr % 16) != 0:
                    fout.write(" 0000")
                    addr +=1
        elif "_F1" in name:
            loops = int((group[2] + 15)/16)
            uncal = group[2]
            fchk = open("check.txt",'w')
            for loop in range(loops):
                #设置输出值
                if uncal >= 16:
                    act = 16
                    full = 1
                else:
                    act = uncal
                    full = 0

                for inode in range(group[1]):
                    #输入点的排列，对应着哪个坐标
                    if inode == 0:
                        uncal_ihor = reshape[0]
                        uncal_iver = reshape[1]
                        uncal_ich  = reshape[2]
                        start_ihor = 0
                        start_iver = 0
                        start_ich  = 0
                        off_hor    = 0
                        off_ver    = 0
                    elif off_hor < min(3,uncal_ihor-1):
                        off_hor +=1
                    elif off_ver < min(3,uncal_iver-1):
                        off_hor = 0
                        off_ver +=1
                    elif uncal_ihor > 4:
                        off_hor = 0
                        off_ver = 0
                        start_ihor +=4
                        uncal_ihor -=4
                    elif uncal_iver > 4:
                        off_hor = 0
                        off_ver = 0
                        start_ihor = 0
                        uncal_ihor = reshape[0]
                        start_iver +=4
                        uncal_iver -=4
                    elif uncal_ich > 1:
                        off_hor = 0
                        off_ver = 0
                        start_ihor = 0
                        uncal_ihor = reshape[0]
                        start_iver = 0
                        uncal_iver = reshape[1]
                        start_ich  += 1
                        uncal_ich  -= 1
                    else:
                        print("Error: loop %d get wrong index!" % inode)
                        quit()
                    hor_idx = start_ihor + off_hor
                    ver_idx = start_iver + off_ver
                    fchk.write("H:%d,V:%d,C:%d\n" % (hor_idx,ver_idx,start_ich))

                    fout.write("\n@%08x" % addr)
                    addr += 16
                    #针对每一个输入计算index
                    for node in range(act):
                        index = group[1] * (loop * 16 + node) + (hor_idx + ver_idx * reshape[0] + start_ich * reshape[1] * reshape[0])
                        num = weights[index].rstrip()
                        fout.write(" " + num)
                    for node in range(16 - act):
                        fout.write(" 0000")
                #bias
                fout.write("\n@%08x" % addr)
                addr += 16
                for node in range(act):
                    index = loop * 16 + node
                    num = bias[index].rstrip()
                    fout.write(" " + num)
                for node in range(16 - act):
                    fout.write(" 0000")
                uncal = uncal - 16
            fchk.close()
        elif "_F" in name:
            loops = int((group[2] + 15)/16)
            uncal = group[2]
            for loop in range(loops):
                if uncal >= 16:
                    act = 16
                    full = 1
                else:
                    act = uncal
                    full = 0
                for inode in range(group[1]):
                    fout.write("\n@%08x" % addr)
                    addr += 16
                    for node in range(act):
                        index = group[1] * (loop * 16 + node) + inode
                        num = weights[index].rstrip()
                        fout.write(" " + num)
                    for node in range(16 - act):
                        fout.write(" 0000")
                #bias
                fout.write("\n@%08x" % addr)
                addr += 16
                for node in range(act):
                    index = loop * 16 + node
                    num = bias[index].rstrip()
                    fout.write(" " + num)
                for node in range(16 - act):
                    fout.write(" 0000")
                uncal = uncal - 16


