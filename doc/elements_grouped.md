## Классы компонентов в программе

**cl_id** - номер класса, использующийся в исходном коде. Одинаковые компоненты, в разных поворотах имеют разный cl_id

**rot** - количество поворотов на 90° против часовой стрелки от базового положения.

**shape** - разрешение изображения, которое заложено в корреляционную функцию.

**name** - имя компонента в программе (можно возвратить, обычно не используется)

**image** - изображение компонента

**pattern** - изображение, которое используется в корреляционной функции для поиска кандидатов

| cl_id | rot  | shape      | name                | image | pattern |
| :---: | :--: | :--------- | :------------------ | :--: |:--:|
|	0	|	0	|	Unknown	|	not elem	|	-	|-|
|	1	|	0	|	Unknown	|	not elem/0	|	-	|-|
|	2	|	0	|	Unknown	|	not elem/1	|	-	|-|
|	3	|	0	|	Unknown	|	not elem/2	|	-	|-|
|	4	|	0	|	Unknown	|	not elem/3	|	-	|-|
|	5	|	0	|	(100, 100)	|	SOT323	| ![](elements\images\SOT323.bmp)	|	![](elements\images_patterns\5.bmp)	|
|	39	|	1	|	(100, 100)	|	SOT323	|![](elements\images\id39.bmp)	|	![](elements\images_patterns\39.bmp)	|
|	40	|	2	|	(100, 100)	|	SOT323	|![](elements\images\id40.bmp)	|	![](elements\images_patterns\40.bmp)	|
|	41	|	3	|	(100, 100)	|	SOT323	|![](elements\images\id41.bmp)	|	![](elements\images_patterns\41.bmp)	|
|	6	|	0	|	(100, 100)	|	SOT323/0	| ![SOT323+0](elements\images\SOT3230.bmp)	|	![](elements\images_patterns\6.bmp)	|
|	42	|	1	|	(100, 100)	|	SOT323/0	|![](elements\images\id42.bmp)	|	![](elements\images_patterns\42.bmp)	|
|	43	|	2	|	(100, 100)	|	SOT323/0	|![](elements\images\id43.bmp)	|	![](elements\images_patterns\43.bmp)	|
|	44	|	3	|	(100, 100)	|	SOT323/0	|![](elements\images\id44.bmp)	|	![](elements\images_patterns\44.bmp)	|
|	7	|	0	|	(64, 100)	|	SOT523	| ![SOT523](elements\images\SOT523.bmp)	|	![](elements\images_patterns\7.bmp)	|
|	45	|	1	|	(100, 64)	|	SOT523	|![](elements\images\id45.bmp)	|	![](elements\images_patterns\45.bmp)	|
|	46	|	2	|	(64, 100)	|	SOT523	|![](elements\images\id46.bmp)	|	![](elements\images_patterns\46.bmp)	|
|	47	|	3	|	(100, 64)	|	SOT523	|![](elements\images\id47.bmp)	|	![](elements\images_patterns\47.bmp)	|
|	8	|	0	|	(100, 100)	|	SOT23-5	| ![SOT23-5](elements\images\SOT23-5.bmp)	|	![](elements\images_patterns\8.bmp)	|
|	48	|	1	|	(100, 100)	|	SOT23-5	|![](elements\images\id48.bmp)	|	![](elements\images_patterns\48.bmp)	|
|	49	|	2	|	(100, 100)	|	SOT23-5	|![](elements\images\id49.bmp)	|	![](elements\images_patterns\49.bmp)	|
|	50	|	3	|	(100, 100)	|	SOT23-5	|![](elements\images\id50.bmp)	|	![](elements\images_patterns\50.bmp)	|
|	9	|	0	|	(68, 76)	|	SOT323-5	| ![SOT323-5](elements\images\SOT323-5.bmp)	|	![](elements\images_patterns\9.bmp)	|
|	51	|	1	|	(76, 68)	|	SOT323-5	|![](elements\images\id51.bmp)	|	![](elements\images_patterns\51.bmp)	|
|	52	|	2	|	(68, 76)	|	SOT323-5	|![](elements\images\id52.bmp)	|	![](elements\images_patterns\52.bmp)	|
|	53	|	3	|	(76, 68)	|	SOT323-5	|![](elements\images\id53.bmp)	|	![](elements\images_patterns\53.bmp)	|
|	10	|	0	|	(100, 100)	|	SOT23-6	| ![SOT23-6](elements\images\SOT23-6.bmp)	|	![](elements\images_patterns\10.bmp)	|
|	54	|	1	|	(100, 100)	|	SOT23-6	|![](elements\images\id54.bmp)	|	![](elements\images_patterns\54.bmp)	|
|	11	|	0	|	(68, 76)	|	SOT363	| ![SOT363](elements\images\SOT363.bmp)	|	![](elements\images_patterns\11.bmp)	|
|	55	|	1	|	(76, 68)	|	SOT363	|![](elements\images\id55.bmp)	|	![](elements\images_patterns\55.bmp)	|
|	12	|	0	|	(68, 76)	|	SOT343	|![](elements\images\SOT343.bmp)	|	![](elements\images_patterns\12.bmp)	|
|	56	|	1	|	(76, 68)	|	SOT343	|![](elements\images\id56.bmp)	|	![](elements\images_patterns\56.bmp)	|
|	13	|	0	|	(100, 92)	|	SOT143	|![](elements\images\SOT143.bmp)	|	![](elements\images_patterns\13.bmp)	|
|	57	|	1	|	(92, 100)	|	SOT143	|![](elements\images\id57.bmp)	|	![](elements\images_patterns\57.bmp)	|
|	14	|	0	|	(54, 54)	|	SOT723	|![](elements\images\SOT723.bmp)	|	![](elements\images_patterns\14.bmp)	|
|	58	|	1	|	(54, 54)	|	SOT723	|![](elements\images\id58.bmp)	|	![](elements\images_patterns\58.bmp)	|
|	59	|	2	|	(54, 54)	|	SOT723	|![](elements\images\id59.bmp)	|	![](elements\images_patterns\59.bmp)	|
|	60	|	3	|	(54, 54)	|	SOT723	|![](elements\images\id60.bmp)	|	![](elements\images_patterns\60.bmp)	|
|	15	|	0	|	(84, 168)	|	SMA	|![](elements\images\SMA.bmp)	|	![](elements\images_patterns\15.bmp)	|
|	61	|	1	|	(168, 84)	|	SMA	|![](elements\images\id61.bmp)	|	![](elements\images_patterns\61.bmp)	|
|	16	|	0	|	(108, 168)	|	SMB	|![](elements\images\SMB.bmp)	|	![](elements\images_patterns\16.bmp)	|
|	62	|	1	|	(168, 108)	|	SMB	|![](elements\images\id62.bmp)	|	![](elements\images_patterns\62.bmp)	|
|	17	|	0	|	(34, 53)	|	2-SMD	|![](elements\images\2-SMD.bmp)	|	![](elements\images_patterns\17.bmp)	|
|	63	|	1	|	(53, 34)	|	2-SMD	|![](elements\images\id63.bmp)	|	![](elements\images_patterns\63.bmp)	|
|	18	|	0	|	(100, 100)	|	SOD110	|![](elements\images\SOD110.bmp)	|	![](elements\images_patterns\18.bmp)	|
|	64	|	1	|	(100, 100)	|	SOD110	|![](elements\images\id64.bmp)	|	![](elements\images_patterns\64.bmp)	|
|	19	|	0	|	(79, 123)	|	SOD323F	|![](elements\images\SOD323F.bmp)	|	![](elements\images_patterns\19.bmp)	|
|	65	|	1	|	(123, 79)	|	SOD323F	|![](elements\images\id65.bmp)	|	![](elements\images_patterns\65.bmp)	|
|	20	|	0	|	(52, 82)	|	SOD523	|![](elements\images\SOD523.bmp)	|	![](elements\images_patterns\20.bmp)	|
|	66	|	1	|	(82, 52)	|	SOD523	|![](elements\images\id66.bmp)	|	![](elements\images_patterns\66.bmp)	|
|	21	|	0	|	(22, 45)	|	SMD0402_CL	|![](elements\images\SMD0402_CL.bmp)	|	![](elements\images_patterns\21.bmp)	|
|	67	|	1	|	(45, 22)	|	SMD0402_CL	|![](elements\images\id67.bmp)	|	![](elements\images_patterns\67.bmp)	|
|	22	|	0	|	(22, 45)	|	SMD0402_R	|![](elements\images\SMD0402_R.bmp)	|	![](elements\images_patterns\22.bmp)	|
|	68	|	1	|	(45, 22)	|	SMD0402_R	|![](elements\images\id68.bmp)	|	![](elements\images_patterns\68.bmp)	|
|	23	|	0	|	(24, 66)	|	SMD0603_CL	|![](elements\images\SMD0603_CL.bmp)	|	![](elements\images_patterns\23.bmp)	|
|	69	|	1	|	(66, 24)	|	SMD0603_CL	|![](elements\images\id69.bmp)	|	![](elements\images_patterns\69.bmp)	|
|	24	|	0	|	(24, 66)	|	SMD0603_R	|![](elements\images\SMD0603_R.bmp)	|	![](elements\images_patterns\24.bmp)	|
|	70	|	1	|	(66, 24)	|	SMD0603_R	|![](elements\images\id70.bmp)	|	![](elements\images_patterns\70.bmp)	|
|	25	|	0	|	(24, 66)	|	SMD0603_R/0	|![](elements\images\SMD0603_R0.bmp)	|	![](elements\images_patterns\25.bmp)	|
|	71	|	1	|	(66, 24)	|	SMD0603_R/0	|![](elements\images\id71.bmp)	|	![](elements\images_patterns\71.bmp)	|
|	26	|	0	|	(24, 66)	|	SMD0603_D	|![](elements\images\SMD0603_D.bmp)	|	![](elements\images_patterns\26.bmp)	|
|	72	|	1	|	(66, 24)	|	SMD0603_D	|![](elements\images\id72.bmp)	|	![](elements\images_patterns\72.bmp)	|
|	27	|	0	|	(36, 78)	|	SMD0805_CL	|![](elements\images\SMD0805_CL.bmp)	|	![](elements\images_patterns\27.bmp)	|
|	73	|	1	|	(78, 36)	|	SMD0805_CL	|![](elements\images\id73.bmp)	|	![](elements\images_patterns\73.bmp)	|
|	28	|	0	|	(36, 78)	|	SMD0805_CL/0	|![](elements\images\SMD0805_CL0.bmp)	|	![](elements\images_patterns\28.bmp)	|
|	74	|	1	|	(78, 36)	|	SMD0805_CL/0	|![](elements\images\id74.bmp)	|	![](elements\images_patterns\74.bmp)	|
|	29	|	0	|	(36, 78)	|	SMD0805_R	|![](elements\images\SMD0805_R.bmp)	|	![](elements\images_patterns\29.bmp)	|
|	75	|	1	|	(78, 36)	|	SMD0805_R	|![](elements\images\id75.bmp)	|	![](elements\images_patterns\75.bmp)	|
|	30	|	0	|	(100, 146)	|	SMD1210_C	|![](elements\images\SMD1210_C.bmp)	|	![](elements\images_patterns\30.bmp)	|
|	76	|	1	|	(146, 100)	|	SMD1210_C	|![](elements\images\id76.bmp)	|	![](elements\images_patterns\76.bmp)	|
|	31	|	0	|	(42, 110)	|	SMD1206_R	|![](elements\images\SMD1206_R.bmp)	|	![](elements\images_patterns\31.bmp)	|
|	77	|	1	|	(110, 42)	|	SMD1206_R	|![](elements\images\id77.bmp)	|	![](elements\images_patterns\77.bmp)	|
|	32	|	0	|	(55, 123)	|	SMD1206_C	|![](elements\images\SMD1206_C.bmp)	|	![](elements\images_patterns\32.bmp)	|
|	78	|	1	|	(123, 55)	|	SMD1206_C	|![](elements\images\id78.bmp)	|	![](elements\images_patterns\78.bmp)	|
|	33	|	0	|	(78, 130)	|	SOIC-%d	|![](elements\images\SOIC-%d.bmp)	|	![](elements\images_patterns\33.bmp)	|
|	79	|	1	|	(130, 78)	|	SOIC-%d	|![](elements\images\id79.bmp)	|	![](elements\images_patterns\79.bmp)	|
|	34	|	0	|	(64, 108)	|	LQFP0.65-%d&SSOP-%d	|![](elements\images\LQFP0.65-%d&SSOP-%d.bmp)	|	![](elements\images_patterns\34.bmp)	|
|	80	|	1	|	(108, 64)	|	LQFP0.65-%d&SSOP-%d	|![](elements\images\id80.bmp)	|	![](elements\images_patterns\80.bmp)	|
|	35	|	0	|	(116, 162)	|	DIP-%d	|![](elements\images\DIP-%d.bmp)	|	![](elements\images_patterns\35.bmp)	|
|	81	|	1	|	(162, 116)	|	DIP-%d	|![](elements\images\id81.bmp)	|	![](elements\images_patterns\81.bmp)	|
|	36	|	0	|	(42, 72)	|	LQFP0.4-%d	|![](elements\images\LQFP0.4-%d.bmp)	|	![](elements\images_patterns\36.bmp)	|
|	82	|	1	|	(72, 42)	|	LQFP0.4-%d	|![](elements\images\id82.bmp)	|	![](elements\images_patterns\82.bmp)	|
|	37	|	0	|	(53, 90)	|	LQFP0.5-%d&TFSOP-%d	|![](elements\images\LQFP0.5-%d&TFSOP-%d.bmp)	|	![](elements\images_patterns\37.bmp)	|
|	83	|	1	|	(90, 53)	|	LQFP0.5-%d&TFSOP-%d	|![](elements\images\id83.bmp)	|	![](elements\images_patterns\83.bmp)	|
|	38	|	0	|	(77, 131)	|	LQFP0.8-%d	|![](elements\images\LQFP0.8-%d.bmp)	|	![](elements\images_patterns\38.bmp)	|
|	84	|	1	|	(131, 77)	|	LQFP0.8-%d	|![](elements\images\id84.bmp)	|	![](elements\images_patterns\84.bmp)	|
