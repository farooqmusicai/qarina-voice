// CI: eSpeak (wahi wasm jo website par chalta hai) se test matn ke ids banata hai
// -> build_piper/ids.json  (python isi ko parhta hai)
import fs from 'fs';
import { makePhonemizer } from './piper_ids.mjs';

const OUT = 'build_piper';
const ESpeakNG = (await import(new URL('../' + OUT + '/espeak-ng.js', import.meta.url).href)).default;
const wasm  = fs.readFileSync(`${OUT}/espeak-ng.wasm`);
const idMap = JSON.parse(fs.readFileSync('models-web/piper_phoneme_ids.json', 'utf8'));

const segs = [
  'قرینہ حاضر ہے۔ یہ اردو زبان کی نئی آواز ہے۔',
  'آسمان جھوٹ نہیں بولتا',
  'دل ناداں تجھے ہوا کیا ہے؟',
  'ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے',
  'بہت نکلے مرے ارمان لیکن پھر بھی کم نکلے',
  'آج موسم بہت اچھا ہے، اور دھوپ نکلی ہوئی ہے۔',
  '1947 میں پاکستان بنا',
];
const ids = await makePhonemizer(ESpeakNG, wasm, idMap)(segs);
segs.forEach((s, i) => console.log(`[${String(ids[i].length).padStart(4)}] ${s}`));
fs.writeFileSync(`${OUT}/ids.json`, JSON.stringify({ segs, ids }));
console.log('ids.json likh diya');
