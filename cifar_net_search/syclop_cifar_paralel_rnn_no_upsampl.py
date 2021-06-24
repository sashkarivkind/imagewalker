'''
The follwing code runs a test lstm network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

cnn_dropout = 0.4 rnn_dropout = 0.2 samples = 10, h = 256, epochs = 150 - out.296107 (Based on best results from cnn - gru)
OVERFITTED like hell
################# parallel_gru_v1_True Validation Accuracy =  [0.41100001335144043, 0.44679999351501465, 0.47760000824928284, 0.4984000027179718, 0.503600001335144, 0.5135999917984009, 0.5249999761581421, 0.5365999937057495, 0.5379999876022339, 0.5590000152587891, 0.5383999943733215, 0.5559999942779541, 0.5586000084877014, 0.5740000009536743, 0.5753999948501587, 0.5723999738693237, 0.5604000091552734, 0.5776000022888184, 0.5821999907493591, 0.5888000130653381, 0.5896000266075134, 0.5825999975204468, 0.5849999785423279, 0.5812000036239624, 0.5946000218391418, 0.5856000185012817, 0.5952000021934509, 0.5842000246047974, 0.5996000170707703, 0.5878000259399414, 0.5795999765396118, 0.5852000117301941, 0.5835999846458435, 0.5889999866485596, 0.5807999968528748, 0.5884000062942505, 0.5870000123977661, 0.5842000246047974, 0.5892000198364258, 0.5834000110626221, 0.5740000009536743, 0.5734000205993652, 0.5794000029563904, 0.571399986743927, 0.5654000043869019, 0.576200008392334, 0.5730000138282776, 0.5716000199317932, 0.5767999887466431, 0.5752000212669373, 0.5730000138282776, 0.573199987411499, 0.5766000151634216, 0.5712000131607056, 0.5669999718666077, 0.5723999738693237, 0.5771999955177307, 0.5789999961853027, 0.5727999806404114, 0.5723999738693237, 0.5659999847412109, 0.5727999806404114, 0.5649999976158142, 0.5691999793052673, 0.569599986076355, 0.5708000063896179, 0.5712000131607056, 0.5774000287055969, 0.5684000253677368, 0.5662000179290771, 0.5753999948501587, 0.5716000199317932, 0.5748000144958496, 0.5727999806404114, 0.5690000057220459, 0.5698000192642212, 0.5752000212669373, 0.5680000185966492, 0.5708000063896179, 0.5752000212669373, 0.5630000233650208, 0.5705999732017517, 0.5716000199317932, 0.5673999786376953, 0.5673999786376953, 0.5690000057220459, 0.5720000267028809, 0.5685999989509583, 0.5720000267028809, 0.5655999779701233, 0.5748000144958496, 0.5623999834060669, 0.5705999732017517, 0.5716000199317932, 0.5740000009536743, 0.58160001039505, 0.576200008392334, 0.571399986743927, 0.5623999834060669, 0.576200008392334, 0.5730000138282776, 0.5777999758720398, 0.5806000232696533, 0.5651999711990356, 0.5741999745368958, 0.5659999847412109, 0.5734000205993652, 0.5723999738693237, 0.5619999766349792, 0.5672000050544739, 0.5698000192642212, 0.5709999799728394, 0.5702000260353088, 0.5631999969482422, 0.5687999725341797, 0.5702000260353088, 0.5812000036239624, 0.5720000267028809, 0.5633999705314636, 0.5795999765396118, 0.5767999887466431, 0.5655999779701233, 0.5759999752044678, 0.5663999915122986, 0.5655999779701233, 0.5681999921798706, 0.5799999833106995, 0.5691999793052673, 0.5654000043869019, 0.5698000192642212, 0.5640000104904175, 0.5758000016212463, 0.5709999799728394, 0.5741999745368958, 0.5770000219345093, 0.578000009059906, 0.5722000002861023, 0.5658000111579895, 0.5702000260353088, 0.5799999833106995, 0.576200008392334, 0.5759999752044678, 0.5703999996185303, 0.5691999793052673, 0.5694000124931335, 0.5633999705314636, 0.5698000192642212, 0.5794000029563904, 0.5785999894142151, 0.5767999887466431]
################# parallel_gru_v1_True Training Accuracy =  [0.3335777819156647, 0.42657777667045593, 0.4543111026287079, 0.4767777919769287, 0.48955556750297546, 0.5038444399833679, 0.5148000121116638, 0.5274222493171692, 0.5319555401802063, 0.5447555780410767, 0.5526221990585327, 0.5611110925674438, 0.5618888735771179, 0.5739777684211731, 0.5810666680335999, 0.5873333215713501, 0.5932888984680176, 0.5970222353935242, 0.6084222197532654, 0.614133358001709, 0.6177555322647095, 0.6252666711807251, 0.6312000155448914, 0.6389333605766296, 0.6492666602134705, 0.6555333137512207, 0.6631555557250977, 0.6762222051620483, 0.6800000071525574, 0.6892889142036438, 0.7003777623176575, 0.7081555724143982, 0.718666672706604, 0.7243777513504028, 0.7342444658279419, 0.7470889091491699, 0.7545555830001831, 0.766177773475647, 0.7746222019195557, 0.7826889157295227, 0.791266679763794, 0.8008221983909607, 0.8044000267982483, 0.8139111399650574, 0.822422206401825, 0.8278444409370422, 0.837755560874939, 0.8401111364364624, 0.8497999906539917, 0.8519999980926514, 0.862155556678772, 0.8641999959945679, 0.8690000176429749, 0.875177800655365, 0.8769333362579346, 0.8816444277763367, 0.8850444555282593, 0.8920888900756836, 0.8951555490493774, 0.8948444724082947, 0.8982444405555725, 0.9026666879653931, 0.9027777910232544, 0.9052666425704956, 0.9073110818862915, 0.9089333415031433, 0.9127777814865112, 0.9121333360671997, 0.9135777950286865, 0.9164666533470154, 0.9193333387374878, 0.9214000105857849, 0.9240000247955322, 0.9215555787086487, 0.9245333075523376, 0.9282888770103455, 0.9275555610656738, 0.9287111163139343, 0.927911102771759, 0.9311333298683167, 0.9294221997261047, 0.9300222396850586, 0.9308221936225891, 0.9391555786132812, 0.9340444207191467, 0.9372888803482056, 0.9377777576446533, 0.932533323764801, 0.9369111061096191, 0.9379777908325195, 0.9412888884544373, 0.9394000172615051, 0.9401777982711792, 0.940488874912262, 0.9404444694519043, 0.9414222240447998, 0.9429333209991455, 0.9446222186088562, 0.9443110823631287, 0.944599986076355, 0.9458444714546204, 0.9471333622932434, 0.9446444511413574, 0.9469777941703796, 0.9457777738571167, 0.9482444524765015, 0.9465555548667908, 0.9499555826187134, 0.9483333230018616, 0.946399986743927, 0.9472666382789612, 0.9499555826187134, 0.9487333297729492, 0.9491999745368958, 0.9537777900695801, 0.9490000009536743, 0.9493333101272583, 0.9517555832862854, 0.9494222402572632, 0.9525555372238159, 0.9527333378791809, 0.9503999948501587, 0.9517777562141418, 0.9536444544792175, 0.951533317565918, 0.9526444673538208, 0.9548444151878357, 0.9569555521011353, 0.9516666531562805, 0.9536444544792175, 0.9544222354888916, 0.9515777826309204, 0.9572666883468628, 0.9551110863685608, 0.9562888741493225, 0.9554888606071472, 0.953000009059906, 0.9552222490310669, 0.957622230052948, 0.9560666680335999, 0.9565555453300476, 0.9568666815757751, 0.9563111066818237, 0.9564889073371887, 0.9604666829109192, 0.9571555852890015, 0.9551777839660645, 0.9562888741493225, 0.9588000178337097, 0.9584222435951233]

cnn_dropout = 0.4 rnn_dropout = 0.2 , parralel_rnn_dropour = 0.4, samples = 10, h = 256, epochs = 50 - out.308216/49 (Based on best results from cnn - gru)
################# parallel_gru_v1_True Validation Accuracy =  [0.41019999980926514, 0.444599986076355, 0.4731999933719635, 0.5, 0.5180000066757202, 0.51419997215271, 0.5356000065803528, 0.5342000126838684, 0.5293999910354614, 0.5550000071525574, 0.5368000268936157, 0.5613999962806702, 0.5640000104904175, 0.5645999908447266, 0.5522000193595886, 0.5627999901771545, 0.58160001039505, 0.5723999738693237, 0.5759999752044678, 0.5735999941825867, 0.5878000259399414, 0.5971999764442444, 0.5879999995231628, 0.5932000279426575, 0.5947999954223633, 0.5821999907493591, 0.5885999798774719, 0.592199981212616, 0.5892000198364258, 0.578000009059906, 0.5896000266075134, 0.5953999757766724, 0.5856000185012817, 0.6025999784469604, 0.5938000082969666, 0.5803999900817871, 0.5813999772071838, 0.5943999886512756, 0.5842000246047974, 0.5907999873161316, 0.5971999764442444, 0.5874000191688538, 0.5838000178337097, 0.58160001039505, 0.5842000246047974, 0.5884000062942505, 0.5781999826431274, 0.5770000219345093, 0.5853999853134155, 0.5881999731063843]
################# parallel_gru_v1_True Training Accuracy =  [0.3397333323955536, 0.4201333224773407, 0.44964444637298584, 0.4718666672706604, 0.4859555661678314, 0.5011110901832581, 0.512844443321228, 0.5214889049530029, 0.5308666825294495, 0.5360222458839417, 0.5474888682365417, 0.5542222261428833, 0.5572222471237183, 0.5652222037315369, 0.5679110884666443, 0.5798888802528381, 0.5851110816001892, 0.5849999785423279, 0.5954444408416748, 0.6005777716636658, 0.6049777865409851, 0.6103333234786987, 0.6199111342430115, 0.6214666962623596, 0.6290222406387329, 0.6344000101089478, 0.6418444514274597, 0.649066686630249, 0.6532666683197021, 0.6614444255828857, 0.667555570602417, 0.6756222248077393, 0.6889111399650574, 0.6902666687965393, 0.7012666463851929, 0.7041777968406677, 0.7142221927642822, 0.7185333371162415, 0.731844425201416, 0.7406222224235535, 0.7424888610839844, 0.7534444332122803, 0.7580888867378235, 0.7674221992492676, 0.7742000222206116, 0.7829777598381042, 0.7888000011444092, 0.7980222105979919, 0.8013777732849121, 0.8088444471359253]

2 parralel chanels (3rd silenced) cnn_dropout = 0.4 rnn_dropout = 0.2 , parralel_rnn_dropour = 0.4, samples = 10, h = 256, epochs = 50 - out.308216/49 (Based on best results from cnn - gru)
out.372496
################# parallel_gru_v1_True Validation Accuracy =  [0.4074000120162964, 0.46059998869895935, 0.4742000102996826, 0.490200012922287, 0.487199991941452, 0.5144000053405762, 0.5285999774932861, 0.5361999869346619, 0.5396000146865845, 0.5631999969482422, 0.5547999739646912, 0.5803999900817871, 0.557200014591217, 0.5637999773025513, 0.5680000185966492, 0.5853999853134155, 0.5738000273704529, 0.5870000123977661, 0.5781999826431274, 0.5756000280380249, 0.5807999968528748, 0.5807999968528748, 0.5807999968528748, 0.5896000266075134, 0.5824000239372253, 0.5961999893188477, 0.592199981212616, 0.6069999933242798, 0.6029999852180481, 0.6051999926567078, 0.5997999906539917, 0.5964000225067139, 0.6043999791145325, 0.5964000225067139, 0.59579998254776, 0.5982000231742859, 0.597000002861023, 0.590399980545044, 0.5961999893188477, 0.5985999703407288, 0.5943999886512756, 0.5907999873161316, 0.5874000191688538, 0.5910000205039978, 0.5861999988555908, 0.5917999744415283, 0.5861999988555908, 0.5888000130653381, 0.5884000062942505, 0.5898000001907349]
################# parallel_gru_v1_True Training Accuracy =  [0.34333333373069763, 0.4271555542945862, 0.45891112089157104, 0.48206666111946106, 0.4966000020503998, 0.5101555585861206, 0.5205777883529663, 0.5291555523872375, 0.5426889061927795, 0.5477111339569092, 0.5547555685043335, 0.5649333596229553, 0.5727999806404114, 0.5781777501106262, 0.5831999778747559, 0.5890222191810608, 0.5941333174705505, 0.6000666618347168, 0.6022666692733765, 0.6122000217437744, 0.6171777844429016, 0.6253555417060852, 0.6274444460868835, 0.6392889022827148, 0.64246666431427, 0.6466888785362244, 0.6546888947486877, 0.662577748298645, 0.6718666553497314, 0.6790000200271606, 0.68586665391922, 0.6950888633728027, 0.7005777955055237, 0.7129555344581604, 0.7156222462654114, 0.7289999723434448, 0.7371777892112732, 0.746844470500946, 0.7544000148773193, 0.7611111402511597, 0.7694000005722046, 0.7785555720329285, 0.7904666662216187, 0.7975999712944031, 0.8029778003692627, 0.8118888735771179, 0.8183333277702332, 0.8254222273826599, 0.834755539894104, 0.8409333229064941]

2 parralel chanels (3rd silenced) cnn_dropout = 0.4 rnn_dropout = 0.2 , parralel_rnn_dropour = 0.4, samples = 20, h = 256, epochs = 50 - out.308216/49 (Based on best results from cnn - gru)
out.372497 23.6.21 WINNER with 62% test accuracy
################# parallel_gru_v1_True Validation Accuracy =  [0.4041999876499176, 0.4553999900817871, 0.483599990606308, 0.5080000162124634, 0.5192000269889832, 0.5257999897003174, 0.5404000282287598, 0.5392000079154968, 0.555400013923645, 0.5651999711990356, 0.545799970626831, 0.578000009059906, 0.5709999799728394, 0.5673999786376953, 0.5807999968528748, 0.5577999949455261, 0.5874000191688538, 0.5839999914169312, 0.6010000109672546, 0.5997999906539917, 0.6015999913215637, 0.6051999926567078, 0.5870000123977661, 0.5992000102996826, 0.6007999777793884, 0.6014000177383423, 0.5947999954223633, 0.6204000115394592, 0.5996000170707703, 0.6050000190734863, 0.6019999980926514, 0.6133999824523926, 0.6087999939918518, 0.6111999750137329, 0.6092000007629395, 0.6141999959945679, 0.6144000291824341, 0.6176000237464905, 0.5946000218391418, 0.6055999994277954, 0.6021999716758728, 0.607200026512146, 0.6133999824523926, 0.6078000068664551, 0.6128000020980835, 0.5992000102996826, 0.5996000170707703, 0.5946000218391418, 0.6011999845504761, 0.5974000096321106]
################# parallel_gru_v1_True Training Accuracy =  [0.34164443612098694, 0.43166667222976685, 0.46915555000305176, 0.49008888006210327, 0.506422221660614, 0.5178222060203552, 0.5307999849319458, 0.5395110845565796, 0.5479111075401306, 0.5608888864517212, 0.5659999847412109, 0.5734666585922241, 0.5775111317634583, 0.585622251033783, 0.5922666788101196, 0.59897780418396, 0.6054888963699341, 0.6096222400665283, 0.6156444549560547, 0.6230666637420654, 0.6311777830123901, 0.6346889138221741, 0.6409555673599243, 0.647422194480896, 0.656155526638031, 0.6636666655540466, 0.6696444153785706, 0.6728444695472717, 0.6814444661140442, 0.6916000247001648, 0.6970221996307373, 0.707622230052948, 0.7156000137329102, 0.7226666808128357, 0.7283555269241333, 0.7379778027534485, 0.7448889017105103, 0.7492444515228271, 0.7555999755859375, 0.766177773475647, 0.7761333584785461, 0.7790444493293762, 0.7900000214576721, 0.7919333577156067, 0.7984889149665833, 0.8065333366394043, 0.8131333589553833, 0.8182222247123718, 0.8252888917922974, 0.8286222219467163]


cnn_dropout = 0.4 rnn_dropout = 0.2 , parralel_rnn_dropour = 0.4, samples = 10, h = 256, epochs = 200 - out.308216/49 (Based on best results from cnn - gru)
with kernel_regularizer keras.regularizers.l1_l2(l1=0.01, l2=0.01) out.439119
################# parallel_gru_v1_True Validation Accuracy =  [0.35280001163482666, 0.41019999980926514, 0.42980000376701355, 0.4603999853134155, 0.4553999900817871, 0.47360000014305115, 0.4797999858856201, 0.4943999946117401, 0.4986000061035156, 0.49480000138282776, 0.5123999714851379, 0.5253999829292297, 0.5192000269889832, 0.5314000248908997, 0.5278000235557556, 0.5401999950408936, 0.5375999808311462, 0.5317999720573425, 0.5342000126838684, 0.5284000039100647, 0.5314000248908997, 0.5486000180244446, 0.5450000166893005, 0.5407999753952026, 0.5333999991416931, 0.5317999720573425, 0.5630000233650208, 0.5432000160217285, 0.5450000166893005, 0.5613999962806702, 0.5568000078201294, 0.5651999711990356, 0.5613999962806702, 0.5580000281333923, 0.5522000193595886, 0.5662000179290771, 0.5577999949455261, 0.5644000172615051, 0.569599986076355, 0.5641999840736389, 0.569599986076355, 0.5662000179290771, 0.571399986743927, 0.5698000192642212, 0.5623999834060669, 0.5633999705314636, 0.576200008392334, 0.5802000164985657, 0.5663999915122986, 0.576200008392334, 0.5776000022888184, 0.5857999920845032, 0.5582000017166138, 0.5842000246047974, 0.5807999968528748, 0.5694000124931335, 0.5799999833106995, 0.5852000117301941, 0.5807999968528748, 0.5856000185012817, 0.5709999799728394, 0.5758000016212463, 0.5924000144004822, 0.5848000049591064, 0.5881999731063843, 0.5794000029563904, 0.5896000266075134, 0.5898000001907349, 0.5848000049591064, 0.5893999934196472, 0.5867999792098999, 0.5727999806404114, 0.5952000021934509, 0.5943999886512756, 0.58160001039505, 0.5874000191688538, 0.5770000219345093, 0.5866000056266785, 0.5776000022888184, 0.5914000272750854, 0.5964000225067139, 0.5878000259399414, 0.5885999798774719, 0.5853999853134155, 0.5852000117301941, 0.5885999798774719, 0.5899999737739563, 0.5853999853134155, 0.5835999846458435, 0.574400007724762, 0.5888000130653381, 0.5857999920845032, 0.5866000056266785, 0.5881999731063843, 0.5878000259399414, 0.5817999839782715, 0.5838000178337097, 0.5875999927520752, 0.5925999879837036, 0.5776000022888184, 0.5985999703407288, 0.5953999757766724, 0.592199981212616, 0.5942000150680542, 0.5924000144004822, 0.5776000022888184, 0.599399983882904, 0.5924000144004822, 0.5925999879837036, 0.6011999845504761, 0.5992000102996826, 0.5978000164031982, 0.5861999988555908, 0.5863999724388123, 0.6092000007629395, 0.5920000076293945, 0.5974000096321106, 0.5992000102996826, 0.5889999866485596, 0.5956000089645386, 0.5956000089645386, 0.5709999799728394, 0.5870000123977661, 0.6010000109672546, 0.5885999798774719, 0.597000002861023, 0.5893999934196472, 0.5860000252723694, 0.5871999859809875, 0.592199981212616, 0.6007999777793884, 0.6029999852180481, 0.5863999724388123, 0.5834000110626221, 0.5871999859809875, 0.5888000130653381, 0.5992000102996826, 0.5906000137329102, 0.6029999852180481, 0.5910000205039978, 0.6055999994277954, 0.5985999703407288, 0.592199981212616, 0.5932000279426575, 0.597000002861023, 0.6010000109672546, 0.5947999954223633, 0.5985999703407288, 0.5928000211715698, 0.5870000123977661, 0.5929999947547913, 0.5996000170707703, 0.6047999858856201, 0.5953999757766724, 0.5964000225067139, 0.597000002861023, 0.5974000096321106, 0.6010000109672546, 0.5996000170707703, 0.5898000001907349, 0.5916000008583069, 0.5989999771118164, 0.5985999703407288, 0.5938000082969666, 0.5974000096321106, 0.5911999940872192, 0.5985999703407288, 0.5924000144004822, 0.5857999920845032, 0.6007999777793884, 0.5924000144004822, 0.599399983882904, 0.6028000116348267, 0.5964000225067139, 0.5928000211715698, 0.603600025177002, 0.5943999886512756, 0.5974000096321106, 0.6007999777793884, 0.6015999913215637, 0.6043999791145325, 0.6114000082015991, 0.604200005531311, 0.602400004863739, 0.6033999919891357, 0.599399983882904, 0.5766000151634216, 0.6000000238418579, 0.5885999798774719, 0.5950000286102295, 0.5910000205039978, 0.5982000231742859, 0.6100000143051147, 0.6019999980926514, 0.5914000272750854, 0.5917999744415283, 0.603600025177002, 0.6075999736785889, 0.5892000198364258, 0.6015999913215637]
################# parallel_gru_v1_True Training Accuracy =  [0.2652444541454315, 0.3714222311973572, 0.40773332118988037, 0.4268888831138611, 0.4393777847290039, 0.4533555507659912, 0.46417778730392456, 0.4739777743816376, 0.4826222360134125, 0.4915555417537689, 0.4980444312095642, 0.5035555362701416, 0.5073555707931519, 0.5148444175720215, 0.52028888463974, 0.5202000141143799, 0.527044415473938, 0.5299555659294128, 0.533466637134552, 0.5370222330093384, 0.5411111116409302, 0.5446444153785706, 0.5469555258750916, 0.5516444444656372, 0.5521555542945862, 0.5523333549499512, 0.5591777563095093, 0.5591999888420105, 0.5596888661384583, 0.5635555386543274, 0.5675333142280579, 0.5686444640159607, 0.5725333094596863, 0.5724666714668274, 0.5746889114379883, 0.5753777623176575, 0.5761333107948303, 0.5784222483634949, 0.5810444355010986, 0.5812888741493225, 0.5853555798530579, 0.5871777534484863, 0.5868666768074036, 0.5879777669906616, 0.5893555283546448, 0.5902888774871826, 0.5931333303451538, 0.5902888774871826, 0.5944888591766357, 0.5952444672584534, 0.5969777703285217, 0.595644474029541, 0.5990222096443176, 0.602222204208374, 0.6000666618347168, 0.6013555526733398, 0.59897780418396, 0.6051333546638489, 0.600933313369751, 0.6072666645050049, 0.6060000061988831, 0.6039555668830872, 0.6079333424568176, 0.6088888645172119, 0.6067110896110535, 0.6104221940040588, 0.6108444333076477, 0.611466646194458, 0.6125555634498596, 0.6129778027534485, 0.6147333383560181, 0.6130444407463074, 0.6178444623947144, 0.61644446849823, 0.6158000230789185, 0.6170666813850403, 0.6192222237586975, 0.6198889017105103, 0.619866669178009, 0.618755578994751, 0.620711088180542, 0.6235777735710144, 0.6237778067588806, 0.622511088848114, 0.6219555735588074, 0.6227555274963379, 0.6244666576385498, 0.6255778074264526, 0.6254444718360901, 0.6272222399711609, 0.6281777620315552, 0.6270444393157959, 0.6298888921737671, 0.6301555633544922, 0.6278444528579712, 0.629288911819458, 0.632111132144928, 0.6273777484893799, 0.6311333179473877, 0.6321555376052856, 0.6328222155570984, 0.6295999884605408, 0.628777801990509, 0.6342666745185852, 0.635200023651123, 0.631600022315979, 0.6331999897956848, 0.6342222094535828, 0.6351555585861206, 0.6334444284439087, 0.637333333492279, 0.6392889022827148, 0.6359555721282959, 0.6354444622993469, 0.6361111402511597, 0.633400022983551, 0.6390222311019897, 0.6380888819694519, 0.6410666704177856, 0.636555552482605, 0.6418889164924622, 0.6397777795791626, 0.6408666372299194, 0.6440444588661194, 0.6383333206176758, 0.6419110894203186, 0.6397555470466614, 0.6410222053527832, 0.6424444317817688, 0.6418222188949585, 0.6394666433334351, 0.6453555822372437, 0.6466000080108643, 0.6448000073432922, 0.6431999802589417, 0.6452444195747375, 0.6475777626037598, 0.6456444263458252, 0.6453555822372437, 0.6437555551528931, 0.6431999802589417, 0.6442221999168396, 0.6460000276565552, 0.6503111124038696, 0.6474444270133972, 0.6511777639389038, 0.6479777693748474, 0.6502000093460083, 0.6472444534301758, 0.651199996471405, 0.6506666541099548, 0.6517778038978577, 0.6481555700302124, 0.6517555713653564, 0.6531333327293396, 0.6484000086784363, 0.6525555849075317, 0.6533555388450623, 0.6490888595581055, 0.6519111394882202, 0.6509110927581787, 0.6501111388206482, 0.651711106300354, 0.6524222493171692, 0.6528666615486145, 0.6499999761581421, 0.6535778045654297, 0.6541555523872375, 0.6520666480064392, 0.6552888751029968, 0.6536444425582886, 0.6563110947608948, 0.6556666493415833, 0.6575999855995178, 0.6525555849075317, 0.6603555679321289, 0.6562666893005371, 0.6598666906356812, 0.6573777794837952, 0.6570000052452087, 0.6573333144187927, 0.6580666899681091, 0.6589999794960022, 0.6555555462837219, 0.6542666554450989, 0.6587111353874207, 0.6596666574478149, 0.6577555537223816, 0.6572222113609314, 0.6594444513320923, 0.6576666831970215, 0.6601555347442627, 0.6603111028671265, 0.6591333150863647, 0.6591333150863647, 0.6586889028549194, 0.6585333347320557, 0.659333348274231, 0.660444438457489, 0.6590666770935059]
same with 500 epochs

'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
import numpy as np
import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras_utils import *
from misc import *

import tensorflow.keras as keras
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy


#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp

def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_AREA)
    return dwnsmp

# import importlib
# importlib.reload(misc)
# from misc import Logger
# import os 


# def deploy_logs():
#     if not os.path.exists(hp.save_path):
#         os.makedirs(hp.save_path)

#     dir_success = False
#     for sfx in range(1):  # todo legacy
#         candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
#         if not os.path.exists(candidate_path):
#             hp.this_run_path = candidate_path
#             os.makedirs(hp.this_run_path)
#             dir_success = Truecnn_net = cnn_one_img(n_timesteps = sample, input_size = 28, input_dim = 1)
#             break
#     if not dir_success:
#         error('run name already exists!')

#     sys.stdout = Logger(hp.this_run_path+'log.log')
#     print('results are in:', hp.this_run_path)
#     print('description: ', hp.description)
#     #print('hyper-parameters (partial):', hp.dict)
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4
    }
    
else:
    paramaters = {
    'epochs' : 1,
    
    'sample' : 5,
    
    'res' : 8,
    
    'hidden_size' : 128,
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4
    }
    
print(paramaters)
for key,val in paramaters.items():
    exec(key + '=val')
n_timesteps = sample
def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def parallel_gru(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True):
    '''
    
    CNN RNN combination that extends the CNN to a network that achieves 
    ~80% accuracy on full res cifar.

    Parameters
    ----------
    n_timesteps : TYPE, optional
        DESCRIPTION. The default is 5.
    img_dim : TYPE, optional
        DESCRIPTION. The default is 32.
    hidden_size : TYPE, optional
        DESCRIPTION. The default is 128.
    input_size : TYPE, optional
        DESCRIPTION. The default is 32.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    
    ###################### CNN Chanell 1#######################################
    
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    
    ###################### Parallel Chanell 1##################################
    rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    if concat:
        rnn_temp = keras.layers.Concatenate()([rnn_temp,inputB])
    else:
        rnn_temp = keras.layers.Concatenate()([rnn_temp])
    print('flat shape after cnn1', rnn_temp.shape)
    rnn_x = keras.layers.GRU( hidden_size,input_shape=(n_timesteps, None),
                             kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                             return_sequences=True,recurrent_dropout=2*rnn_dropout,
                             )(rnn_temp)
    print('gru hidden states 1 ', rnn_x.shape)
    ###################### CNN Chanell 2 #######################################
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2),name = 'test'),name = 'test')(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    
    ###################### Parallel Chanell 2 ##################################
    rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print('flat shape after cnn2',rnn_temp.shape)  
    if concat:
        rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp,inputB])
    else:
        rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp])
    print(' cnn2 input combined with fst hidden state', rnn_temp.shape)
    rnn_x = keras.layers.GRU( hidden_size,input_shape=(n_timesteps, None),
                             kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                             return_sequences=True,recurrent_dropout=2*rnn_dropout,
                             )(rnn_temp)
    print('gru hidden states 2 ', rnn_x.shape)
    
    ###################### CNN Chanell 3 #######################################
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    print(x1.shape)
    
    ###################### Parallel Chanell 3 ##################################
    # rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    # print('flat shape after cnn3',rnn_temp.shape)
    # if concat:
    #     rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp,inputB])
    # else:
    #     rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp])
    # print(' cnn23input combined with snd hidden state', rnn_temp.shape)
    # rnn_x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=2*rnn_dropout)(rnn_temp)
    # print('gru hidden states 3 ', rnn_x.shape)
    
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)

    if concat:
        x = keras.layers.Concatenate()([x1,rnn_x,inputB])
    else:
        x = keras.layers.Concatenate()([x1,rnn_x])
    print(x.shape)

    # define LSTM model
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=rnn_dropout)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'parallel_gru_v1_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = parallel_gru(n_timesteps = sample, hidden_size = hidden_size,input_size = res, concat = True)
#keras.utils.plot_model(rnn_net, expand_nested=True,  to_file='{}.png'.format(rnn_net.name))
#cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = res, dropout = cnn_dropout)
#%%

# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop cifar net search runs"
# hp.this_run_name = 'syclop_{}'.format(rnn_net.name)
# deploy_logs()

train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )
                                    #bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
#%%
# sample = (train_dataset_x[0][40:42],train_dataset_x[1][40:42])
# layer_name = 'test'
# intermediate_layer_model = keras.Model(inputs=rnn_net.input,
#                                        outputs=rnn_net.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model(sample)
# print(np.max(intermediate_output),np.min(intermediate_output),np.mean(intermediate_output),np.std(intermediate_output))
# layer_name = 'test_rnn'
# intermediate_layer_model = keras.Model(inputs=rnn_net.input,
#                                        outputs=rnn_net.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model(sample)
# print(np.max(intermediate_output),np.min(intermediate_output),np.mean(intermediate_output),np.std(intermediate_output))
#%%
print("##################### Fit {} and trajectories model on training data res = {} ##################".format(rnn_net.name,res))
rnn_history = rnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 0)

#%%
# sample = (train_dataset_x[0][40:42],train_dataset_x[1][40:42])
# layer_name = 'test'
# intermediate_layer_model = keras.Model(inputs=rnn_net.input,
#                                        outputs=rnn_net.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model(sample)
# print(np.max(intermediate_output),np.min(intermediate_output),np.mean(intermediate_output),np.std(intermediate_output))
# layer_name = 'test_rnn'
# intermediate_layer_model = keras.Model(inputs=rnn_net.input,
#                                        outputs=rnn_net.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model(sample)
# print(np.max(intermediate_output),np.min(intermediate_output),np.mean(intermediate_output),np.std(intermediate_output))
#%%
print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
# plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
# plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
plt.legend()
plt.title('{} on cifar res = {} hs = {} dropout = {}, num samples = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout,sample))
plt.savefig('{} on Cifar res = {}, no upsample, val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict{}_{}'.format(rnn_net.name, hidden_size,cnn_dropout), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
# with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict'.format(cnn_net.name), 'wb') as file_pi:
#     pickle.dump(cnn_history.history, file_pi)
    