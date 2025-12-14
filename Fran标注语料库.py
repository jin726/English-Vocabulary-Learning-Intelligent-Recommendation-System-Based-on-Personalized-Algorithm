import re
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd


@dataclass
class Annotation:
    """标注记录数据类"""
    id: int = None
    chapter_num: str = ""
    narrator: str = ""
    appellation: str = ""
    original_word: str = ""
    context: str = ""
    behavior: str = ""
    naming_conflict: str = "否"
    quality_note: str = ""
    position: int = 0
    timestamp: str = ""


class TextProcessor:
    """文本处理模块"""

    def __init__(self, file_path: str):
        """初始化文本处理器"""
        self.file_path = file_path
        self.text = self._load_text()
        self.chapters = {}

    def _load_text(self) -> str:
        """加载文本文件"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✓ 成功加载文本，大小: {len(content)} 字符")
                return content
        except FileNotFoundError:
            print(f"❌ 文件未找到: {self.file_path}")
            return ""
        except Exception as e:
            print(f"❌ 读取文件错误: {e}")
            return ""

    def analyze_text_structure(self) -> Dict:
        """分析文本结构"""
        stats = {
            'total_chars': len(self.text),
            'total_lines': len(self.text.split('\n')),
            'total_paragraphs': len(self.text.split('\n\n')),
            'has_chapter_marks': bool(re.search(r'(Chapter|CHAPTER|chapter)\s+\d+', self.text)),
            'chapter_patterns': self._find_chapter_patterns()
        }
        return stats

    def _find_chapter_patterns(self) -> List[str]:
        """找出所有可能的章节标记"""
        patterns = []

        # 查找Chapter标记
        chapter_matches = re.findall(r'(Chapter|CHAPTER|chapter)\s+([0-9IVivx]+)', self.text)
        if chapter_matches:
            patterns.extend([f"{m[0]} {m[1]}" for m in chapter_matches[:5]])

        # 查找其他标记
        other_marks = re.findall(r'^[A-Z\s]{20,}$', self.text, re.MULTILINE)
        if other_marks:
            patterns.extend(other_marks[:5])

        return patterns


class SmartChunkDivider:
    """智能分章模块"""

    def __init__(self, text: str):
        """初始化分章器"""
        self.text = text
        self.chapters = {}

    def split_chapters(self) -> Dict[str, str]:
        """按多种方式分割章节"""

        # 方法1: 尝试按Chapter标记分割
        chapters = self._split_by_chapter_marks()
        if chapters and len(chapters) > 1:
            print(f"✓ 使用方法1（Chapter标记）分割为 {len(chapters)} 章")
            return chapters

        # 方法2: 按标题行分割
        chapters = self._split_by_section_headers()
        if chapters and len(chapters) > 1:
            print(f"✓ 使用方法2（标题行）分割为 {len(chapters)} 章")
            return chapters

        # 方法3: 按内容大小分割
        chapters = self._split_by_content_size()
        print(f"✓ 使用方法3（内容大小）分割为 {len(chapters)} 个部分")
        return chapters

    def _split_by_chapter_marks(self) -> Dict[str, str]:
        """按Chapter标记分割"""
        chapters = {}
        pattern = r'(?:^|\n)(CHAPTER|Chapter|chapter)\s+([0-9]+|[IVivx]+)'

        matches = list(re.finditer(pattern, self.text))

        if not matches:
            return {}

        for i, match in enumerate(matches):
            chapter_label = match.group(2)
            chapter_key = f"第{chapter_label}章"

            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(self.text)

            chapter_text = self.text[start_pos:end_pos].strip()
            if chapter_text:
                chapters[chapter_key] = chapter_text

        self.chapters = chapters
        return chapters

    def _split_by_section_headers(self) -> Dict[str, str]:
        """按标题行分割"""
        chapters = {}

        # 找出大写标题行
        header_pattern = r'^[A-Z][A-Z\s]{10,}$'
        lines = self.text.split('\n')

        header_indices = []
        for i, line in enumerate(lines):
            if re.match(header_pattern, line.strip()) and len(line.strip()) > 5:
                header_indices.append((i, line.strip()))

        if len(header_indices) < 2:
            return {}

        for idx, (header_idx, header_text) in enumerate(header_indices):
            chapter_key = f"第{idx + 1}章"

            start_line = header_idx + 1
            end_line = header_indices[idx + 1][0] if idx + 1 < len(header_indices) else len(lines)

            chapter_text = '\n'.join(lines[start_line:end_line]).strip()
            if chapter_text:
                chapters[chapter_key] = chapter_text

        self.chapters = chapters
        return chapters

    def _split_by_content_size(self) -> Dict[str, str]:
        """按内容大小分割"""
        chapters = {}

        # 按段落分割
        paragraphs = self.text.split('\n\n')

        current_section = ""
        section_num = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            current_section += para + "\n\n"

            # 每3000字符创建一个新章节
            if len(current_section) > 3000:
                chapter_key = f"第{section_num}节"
                chapters[chapter_key] = current_section.strip()
                current_section = ""
                section_num += 1

        if current_section.strip():
            chapter_key = f"第{section_num}节"
            chapters[chapter_key] = current_section.strip()

        self.chapters = chapters
        return chapters

    def get_chapters(self) -> Dict[str, str]:
        """获取所有章节"""
        return self.chapters


class AppellationLibrary:
    """称呼词汇库"""

    def __init__(self):
        """初始化称呼库（中英对照）"""
        self.appellations = {
            '怪物': {
                'en_keywords': ['monster', 'creature', 'hideous', 'abomination', 'wretch'],
                'cn_keywords': ['怪物', '生物', '怪兽'],
                'variants': ['monsters', 'monstrous', 'monstrosity', 'wretched']
            },
            '恶魔': {
                'en_keywords': ['devil', 'demon', 'fiend', 'evil', 'evil spirit'],
                'cn_keywords': ['恶魔', '魔鬼', '妖魔'],
                'variants': ['devilish', 'demonic', 'devilry', 'satanic']
            },
            '造物': {
                'en_keywords': ['creation', 'being', 'offspring', 'creature', 'made'],
                'cn_keywords': ['造物', '生物', '创造物'],
                'variants': ['created', 'creator', 'created being']
            },
            '不幸者': {
                'en_keywords': ['wretched', 'miserable', 'unfortunate', 'wretch'],
                'cn_keywords': ['不幸者', '可怜人', '不幸的'],
                'variants': ['wretchedly', 'wretchedness', 'misery']
            },
            '亚当': {
                'en_keywords': ['adam', 'first man'],
                'cn_keywords': ['亚当', '第一个人'],
                'variants': ['adam']
            },
            '撒旦': {
                'en_keywords': ['satan', 'lucifer'],
                'cn_keywords': ['撒旦', '路西法'],
                'variants': ['satanic', 'satanical']
            },
            '幽灵': {
                'en_keywords': ['ghost', 'specter', 'phantom', 'spirit'],
                'cn_keywords': ['幽灵', '鬼魂', '幻影'],
                'variants': ['ghostly', 'spectral', 'spectral']
            },
            '诅咒': {
                'en_keywords': ['curse', 'cursed', 'accursed', 'damned', 'curse word'],
                'cn_keywords': ['诅咒', '被诅咒', '诅咒'],
                'variants': ['cursing', 'accursed', 'damnation']
            }
        }

    def get_all_keywords(self) -> Dict[str, List[str]]:
        """获取所有关键词"""
        keywords_dict = {}
        for appellation, info in self.appellations.items():
            all_keywords = info['en_keywords'] + info['cn_keywords'] + info['variants']
            keywords_dict[appellation] = all_keywords
        return keywords_dict

    def find_appellations(self, text: str, context_size: int = 150) -> List[Dict]:
        """在文本中查找所有称呼"""
        results = []

        for appellation, word_info in self.appellations.items():
            all_keywords = (word_info['en_keywords'] +
                            word_info['cn_keywords'] +
                            word_info['variants'])

            for keyword in all_keywords:
                # 创建单词边界正则表达式
                pattern = rf'\b{re.escape(keyword)}\b'

                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # 提取语境
                    start = max(0, match.start() - context_size)
                    end = min(len(text), match.end() + context_size)
                    context = text[start:end].strip()

                    # 清理语境
                    context = ' '.join(context.split())

                    results.append({
                        'appellation': appellation,
                        'original_word': keyword,
                        'position': match.start(),
                        'context': context[:200]
                    })

        # 去重并排序
        results = self._deduplicate(results)
        return results

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """去除重复标注"""
        deduplicated = []
        last_position = -500

        for result in results:
            if result['position'] - last_position > 100:
                deduplicated.append(result)
                last_position = result['position']

        return sorted(deduplicated, key=lambda x: x['position'])


class NarratorRecognizer:
    """叙事者识别模块"""

    def __init__(self):
        """初始化叙事者库"""
        self.narrators = {
            '沃尔顿': {
                'keywords': ['walton', 'letter', 'dear margaret', 'my dear friend',
                             'arctic', 'expedition', 'polar'],
                'cn_keywords': ['沃尔顿', '信件']
            },
            '弗兰肯斯坦': {
                'keywords': ['victor', 'frankenstein', 'myself', 'my childhood', 'my father',
                             'geneva', 'natural philosophy', 'science'],
                'cn_keywords': ['弗兰肯斯坦', '维克多', '我的']
            },
            '造物': {
                'keywords': ['creature', 'i am', 'my creator', 'my father',
                             'alone', 'rejected', 'solitude', 'wretchedness'],
                'cn_keywords': ['造物', '创造物', '我是']
            },
            '其他': {
                'keywords': ['elizabeth', 'henry', 'father', 'mother', 'william',
                             'caroline', 'justine'],
                'cn_keywords': ['伊丽莎白', '亨利', '父亲', '母亲']
            }
        }

    def detect(self, context: str) -> str:
        """检测叙事者"""
        context_lower = context.lower()

        scores = {}
        for narrator, info in self.narrators.items():
            keywords = info['keywords'] + info['cn_keywords']
            score = sum(1 for kw in keywords if kw.lower() in context_lower)
            scores[narrator] = score

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return '未知'


class BehaviorDetector:
    """伴随行为检测模块"""

    def __init__(self):
        """初始化行为库"""
        self.behaviors = {
            '排斥': {
                'keywords': ['reject', 'refuse', 'cast off', 'abandon', 'shun',
                             'repel', 'turn away', 'scorn'],
                'cn_keywords': ['拒绝', '抛弃', '排斥']
            },
            '攻击': {
                'keywords': ['attack', 'pursue', 'strike', 'beat', 'kill', 'destroy',
                             'violence', 'assault', 'hunt'],
                'cn_keywords': ['攻击', '追杀', '暴力']
            },
            '同情': {
                'keywords': ['pity', 'sympathize', 'understand', 'compassion', 'mercy',
                             'forgive', 'love', 'care'],
                'cn_keywords': ['同情', '怜悯', '理解']
            },
            '恐惧': {
                'keywords': ['fear', 'terrified', 'afraid', 'horror', 'dread',
                             'frightened', 'terror', 'panic'],
                'cn_keywords': ['恐惧', '害怕', '惊恐']
            },
            '愤怒': {
                'keywords': ['angry', 'rage', 'furious', 'wrath', 'hatred', 'despise',
                             'detest', 'abhor'],
                'cn_keywords': ['愤怒', '仇恨', '怨恨']
            },
            '独白': {
                'keywords': ['i think', 'i feel', 'my thoughts', 'to myself',
                             'in my heart', 'my mind', 'reflection'],
                'cn_keywords': ['我想', '我觉得', '内心']
            },
            '对话': {
                'keywords': ['said', 'replied', 'asked', 'spoke', 'told', 'exclaimed',
                             '"', "'"],
                'cn_keywords': ['说', '问', '答']
            }
        }

    def detect(self, context: str) -> str:
        """检测伴随行为"""
        context_lower = context.lower()

        detected = []
        for behavior, info in self.behaviors.items():
            keywords = info['keywords'] + info['cn_keywords']
            for keyword in keywords:
                if keyword.lower() in context_lower:
                    detected.append(behavior)
                    break

        return ','.join(detected) if detected else '未标注'


class ConflictAnalyzer:
    """命名冲突分析模块"""

    def __init__(self):
        """初始化冲突检测"""
        self.conflict_patterns = {
            '身份冲突': {
                'pattern': r'(adam|i am|myself).*?(monster|devil|creature|demon|wretch)',
                'description': '自我认可与他者妖魔化的对立'
            },
            '人性冲突': {
                'pattern': r'(human|man|being|person).*?(creature|wretch|abomination|monster)',
                'description': '人性认可与人性否定的对立'
            },
            '创造者冲突': {
                'pattern': r'(creator|father|made).*?(cursed|damned|evil|reject)',
                'description': '创造责任与创造者的伦理冲突'
            },
            '自我认知冲突': {
                'pattern': r'(i am|i feel|my nature).*?(monster|evil|good|wretch)',
                'description': '自我评价与外界评价的矛盾'
            }
        }

    def analyze(self, context: str, narrator: str) -> Tuple[str, str]:
        """分析命名冲突"""
        context_lower = context.lower()

        for conflict_type, pattern_info in self.conflict_patterns.items():
            if re.search(pattern_info['pattern'], context_lower, re.IGNORECASE):
                return '是', f"{conflict_type}: {pattern_info['description']}"

        return '否', ''


class AnnotationDB:
    """标注数据库管理"""

    def __init__(self, db_path: str = 'frankenstein_annotation.db'):
        """初始化数据库"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._init_db()

    def _init_db(self):
        """初始化数据库连接与表"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

            create_table_sql = '''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_num TEXT NOT NULL,
                narrator TEXT NOT NULL,
                appellation TEXT NOT NULL,
                original_word TEXT,
                context TEXT,
                behavior TEXT,
                naming_conflict TEXT,
                quality_note TEXT,
                position INTEGER,
                timestamp TEXT
            )
            '''

            self.cursor.execute(create_table_sql)
            self.conn.commit()
            print(f"✓ 数据库初始化成功: {self.db_path}")

        except sqlite3.Error as e:
            print(f"❌ 数据库初始化失败: {e}")

    def insert(self, annotation: Annotation) -> int:
        """插入单条记录"""
        annotation.timestamp = datetime.now().isoformat()

        try:
            insert_sql = '''
            INSERT INTO annotations 
            (chapter_num, narrator, appellation, original_word, context, behavior,
             naming_conflict, quality_note, position, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''

            self.cursor.execute(insert_sql, (
                annotation.chapter_num,
                annotation.narrator,
                annotation.appellation,
                annotation.original_word,
                annotation.context,
                annotation.behavior,
                annotation.naming_conflict,
                annotation.quality_note,
                annotation.position,
                annotation.timestamp
            ))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"❌ 插入失败: {e}")
            return -1

    def insert_batch(self, annotations: List[Annotation]) -> int:
        """批量插入"""
        count = 0
        for annotation in annotations:
            if self.insert(annotation) > 0:
                count += 1
        return count

    def get_all(self) -> pd.DataFrame:
        """获取所有标注"""
        try:
            query = 'SELECT * FROM annotations ORDER BY chapter_num, position'
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return pd.DataFrame()

    def get_conflicts(self) -> pd.DataFrame:
        """获取命名冲突"""
        try:
            query = "SELECT * FROM annotations WHERE naming_conflict = '是' ORDER BY chapter_num"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return pd.DataFrame()

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {}

        try:
            self.cursor.execute('SELECT COUNT(*) FROM annotations')
            stats['总标注数'] = self.cursor.fetchone()[0]

            self.cursor.execute('''
                SELECT narrator, COUNT(*) FROM annotations 
                GROUP BY narrator ORDER BY COUNT(*) DESC
            ''')
            stats['叙事者分布'] = dict(self.cursor.fetchall())

            self.cursor.execute('''
                SELECT appellation, COUNT(*) FROM annotations 
                GROUP BY appellation ORDER BY COUNT(*) DESC
            ''')
            stats['称呼频率'] = dict(self.cursor.fetchall())

            self.cursor.execute("SELECT COUNT(*) FROM annotations WHERE naming_conflict = '是'")
            stats['命名冲突数'] = self.cursor.fetchone()[0]

            self.cursor.execute('SELECT COUNT(DISTINCT chapter_num) FROM annotations')
            stats['章节数'] = self.cursor.fetchone()[0]

        except sqlite3.Error as e:
            print(f"❌ 统计失败: {e}")

        return stats

    def export_excel(self, output_file: str = 'frankenstein_annotation.xlsx'):
        """导出到Excel"""
        print(f"导出Excel: {output_file}...", end=' ')

        df = self.get_all()
        df_conflict = self.get_conflicts()

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='全部标注', index=False)
            df_conflict.to_excel(writer, sheet_name='命名冲突', index=False)

            from openpyxl.styles import PatternFill, Font, Alignment

            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]

                header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
                header_font = Font(bold=True, color='FFFFFF')

                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center', wrap_text=True)

                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)

        print("✓")

    def export_csv(self, output_file: str = 'frankenstein_annotation.csv'):
        """导出CSV"""
        print(f"导出CSV: {output_file}...", end=' ')
        df = self.get_all()
        df.to_csv(output_file, index=False, encoding='utf-8')
        print("✓")

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()


class AnnotationPipeline:
    """完整标注流水线"""

    def __init__(self, text_path: str):
        """初始化流水线"""
        self.text_path = text_path

        # 初始化各模块
        print("初始化标注系统...")
        self.processor = TextProcessor(text_path)
        self.divider = SmartChunkDivider(self.processor.text)
        self.library = AppellationLibrary()
        self.narrator_recognizer = NarratorRecognizer()
        self.behavior_detector = BehaviorDetector()
        self.conflict_analyzer = ConflictAnalyzer()
        self.database = AnnotationDB()

    def run(self):
        """执行完整标注"""
        print("\n" + "=" * 70)
        print("开始标注流程...")
        print("=" * 70)

        # 1. 分析文本结构
        print("\n[1/5] 分析文本结构...")
        stats = self.processor.analyze_text_structure()
        print(f"  文本大小: {stats['total_chars']} 字符")
        print(f"  总行数: {stats['total_lines']}")
        print(f"  是否包含Chapter标记: {stats['has_chapter_marks']}")
        if stats['chapter_patterns']:
            print(f"  发现的章节标记: {stats['chapter_patterns'][:3]}")

        # 2. 分章
        print("\n[2/5] 执行文本分章...")
        chapters = self.divider.split_chapters()
        print(f"  成功分割为 {len(chapters)} 章")

        # 3. 逐章标注
        print("\n[3/5] 逐章标注...")
        annotations = []
        for chapter_key, chapter_text in chapters.items():
            # 跳过过短的章节
            if len(chapter_text) < 50:
                continue

            print(f"  处理 {chapter_key}...", end=' ')

            # 查找称呼
            appellation_matches = self.library.find_appellations(chapter_text)

            for match in appellation_matches:
                # 检测叙事者
                narrator = self.narrator_recognizer.detect(match['context'])

                # 检测行为
                behavior = self.behavior_detector.detect(match['context'])

                # 检测冲突
                is_conflict, conflict_note = self.conflict_analyzer.analyze(
                    match['context'], narrator
                )

                # 创建标注
                annotation = Annotation(
                    chapter_num=chapter_key,
                    narrator=narrator,
                    appellation=match['appellation'],
                    original_word=match['original_word'],
                    context=match['context'],
                    behavior=behavior,
                    naming_conflict=is_conflict,
                    quality_note=conflict_note,
                    position=match['position']
                )

                annotations.append(annotation)

            print(f"找到 {len(appellation_matches)} 条")

        # 4. 存储到数据库
        print(f"\n[4/5] 存储到数据库...", end=' ')
        inserted = self.database.insert_batch(annotations)
        print(f"插入 {inserted} 条")

        # 5. 统计与导出
        print(f"\n[5/5] 统计与导出...", end=' ')
        stats = self.database.get_statistics()
        self._print_statistics(stats)

        self.database.export_excel('frankenstein_annotation.xlsx')
        self.database.export_csv('frankenstein_annotation.csv')

        print("\n" + "=" * 70)
        print("✓ 标注完成！")
        print("=" * 70)
        print("\n生成的文件:")
        print("  1. frankenstein_annotation.db   (SQLite数据库)")
        print("  2. frankenstein_annotation.xlsx (Excel表格)")
        print("  3. frankenstein_annotation.csv  (CSV文件)")

    def _print_statistics(self, stats: Dict):
        """打印统计信息"""
        print("\n【标注统计结果】")
        print(f"总标注数: {stats.get('总标注数', 0)}")
        print(f"涉及章节: {stats.get('章节数', 0)}")
        print(f"命名冲突: {stats.get('命名冲突数', 0)}")

        if stats.get('叙事者分布'):
            print("\n叙事者分布:")
            for narrator, count in stats['叙事者分布'].items():
                print(f"  {narrator:12} : {count:3} 条")

        if stats.get('称呼频率'):
            print("\n称呼频率:")
            for appellation, count in stats['称呼频率'].items():
                print(f"  {appellation:12} : {count:3} 条")

    def close(self):
        """关闭资源"""
        self.database.close()


# ============ 主程序 ============

if __name__ == '__main__':
    # 文件路径
    text_file = 'Frankenstein.txt'

    # 检查文件
    if not Path(text_file).exists():
        print(f"❌ 文件未找到: {text_file}")
        print("请确保在工作目录中有 Frankenstein.txt 文件")
        exit(1)

    try:
        # 创建流水线
        pipeline = AnnotationPipeline(text_file)

        # 执行标注
        pipeline.run()

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            pipeline.close()
        except:
            pass