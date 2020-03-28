import struct
import io

height = -1
width = -1
####  url= https://stackoverflow.com/questions/5851607/python3-parsing-jpeg-dimension-info
dafile = open('./images/ev0.jpg', 'rb')
filebyte=dafile.read()
jpeg = io.BytesIO(filebyte)
try:

    type_check = jpeg.read(2)
    if type_check != b'\xff\xd8':
      print("Not a JPG")
    else:
      byte = jpeg.read(1)
      k=0
      while byte != b"":
        
        while byte != b'\xff':
             byte = jpeg.read(1)
          
        while byte == b'\xff':
             byte = jpeg.read(1)
             print(ord(byte))
        if (byte >= b'\xC0' and byte <= b'\xC3'):
          a=jpeg.read(3)
          

          j=k
          
        #  h, w = struct.unpack('>HH', jpeg.read(4))
          h,w=struct.unpack('>HH',jpeg.read(4))
          
          break
        else:
          #print('11xx entered')  
          
          jpeg.read(int(struct.unpack(">H", jpeg.read(2))[0])-2)
        k+=1
        byte = jpeg.read(1)

      width = int(w)
      height = int(h)

      print("Width: %s, Height: %s" % (width, height))
finally:
    jpeg.close()

#ass=b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xe1\x03\x94Exif\x00\x00II*\x00\x08\x00\x00\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x80\r\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x00\x12\x00\x00\x0f\x01\x02\x00\x07\x00\x00\x00\xae\x00\x00\x00\x10\x01\x02\x00\r\x00\x00\x00\xb6\x00\x00\x00\x12\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\x9e\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\xa6\x00\x00\x00(\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x002\x01\x02\x00\x14\x00\x00\x00\xc4\x00\x00\x00\x13\x02\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00i\x87\x04\x00\x01\x00\x00\x00\xd8\x00\x00\x00%\x88\x04\x00\x01\x00\x00\x00\xf0\x02\x00\x00\x00\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00realme\x00\x00realme 3 Pro\x00\x002020:03:26 19:25:13\x00 \x00\x01\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x9a\x82\x05\x00\x01\x00\x00\x00\xb8\x02\x00\x00\x9d\x82\x05\x00\x01\x00\x00\x00\xb0\x02\x00\x00"\x88\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\'\x88\x03\x00\x01\x00\x00\x00\x06\x06\x00\x00\x00\x90\x07\x00\x04\x00\x00\x000210\x03\x90\x02\x00\x14\x00\x00\x00^\x02\x00\x00\x04\x90\x02\x00\x14\x00\x00\x00r\x02\x00\x00\x01\x91\x07\x00\x04\x00\x00\x00\x01\x02\x03\x00\x01\x92\x05\x00\x01\x00\x00\x00\xd8\x02\x00\x00\x02\x92\x05\x00\x01\x00\x00\x00\xc0\x02\x00\x00\x03\x92\x05\x00\x01\x00\x00\x00\xe0\x02\x00\x00\x04\x92\n\x00\x01\x00\x00\x00\xc8\x02\x00\x00\x05\x92\x05\x00\x01\x00\x00\x00\xe8\x02\x00\x00\x07\x92\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x92\x03\x00\x01\x00\x00\x00\x10\x00\x00\x00\n\x92\x05\x00\x01\x00\x00\x00\xd0\x02\x00\x00|\x92\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x86\x92\x07'
full=b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xe1\x03\x94Exif\x00\x00II*\x00\x08\x00\x00\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x80\r\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x00\x12\x00\x00\x0f\x01\x02\x00\x07\x00\x00\x00\xae\x00\x00\x00\x10\x01\x02\x00\r\x00\x00\x00\xb6\x00\x00\x00\x12\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\x9e\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\xa6\x00\x00\x00(\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x002\x01\x02\x00\x14\x00\x00\x00\xc4\x00\x00\x00\x13\x02\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00i\x87\x04\x00\x01\x00\x00\x00\xd8\x00\x00\x00%\x88\x04\x00\x01\x00\x00\x00\xf0\x02\x00\x00\x00\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00realme\x00\x00realme 3 Pro\x00\x002020:03:26 19:25:13\x00 \x00\x01\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x9a\x82\x05\x00\x01\x00\x00\x00\xb8\x02\x00\x00\x9d\x82\x05\x00\x01\x00\x00\x00\xb0\x02\x00\x00"\x88\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\'\x88\x03\x00\x01\x00\x00\x00\x06\x06\x00\x00\x00\x90\x07\x00\x04\x00\x00\x000210\x03\x90\x02\x00\x14\x00\x00\x00^\x02\x00\x00\x04\x90\x02\x00\x14\x00\x00\x00r\x02\x00\x00\x01\x91\x07\x00\x04\x00\x00\x00\x01\x02\x03\x00\x01\x92\x05\x00\x01\x00\x00\x00\xd8\x02\x00\x00\x02\x92\x05\x00\x01\x00\x00\x00\xc0\x02\x00\x00\x03\x92\x05\x00\x01\x00\x00\x00\xe0\x02\x00\x00\x04\x92\n\x00\x01\x00\x00\x00\xc8\x02\x00\x00\x05\x92\x05\x00\x01\x00\x00\x00\xe8\x02\x00\x00\x07\x92\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x92\x03\x00\x01\x00\x00\x00\x10\x00\x00\x00\n\x92\x05\x00\x01\x00\x00\x00\xd0\x02\x00\x00|\x92\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x86\x92\x07\x00\x11\x00\x00\x00\x9e\x02\x00\x00\x90\x92\x02\x00\x07\x00\x00\x00\x86\x02\x00\x00'

print(j)