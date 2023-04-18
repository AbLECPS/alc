package edu.vanderbilt.isis.alc.btree.parser.antlr.internal;

import org.eclipse.xtext.*;
import org.eclipse.xtext.parser.*;
import org.eclipse.xtext.parser.impl.*;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.xtext.parser.antlr.AbstractInternalAntlrParser;
import org.eclipse.xtext.parser.antlr.XtextTokenStream;
import org.eclipse.xtext.parser.antlr.XtextTokenStream.HiddenTokens;
import org.eclipse.xtext.parser.antlr.AntlrDatatypeRuleToken;
import edu.vanderbilt.isis.alc.btree.services.BTreeGrammarAccess;



import org.antlr.runtime.*;
import java.util.Stack;
import java.util.List;
import java.util.ArrayList;

@SuppressWarnings("all")
public class InternalBTreeParser extends AbstractInternalAntlrParser {
    public static final String[] tokenNames = new String[] {
        "<invalid>", "<EOR>", "<DOWN>", "<UP>", "RULE_ID", "RULE_INT", "RULE_STRING", "RULE_BOOLEAN", "RULE_ML_COMMENT", "RULE_SL_COMMENT", "RULE_WS", "RULE_ANY_OTHER", "'system'", "';'", "'tree'", "'('", "'updatetime'", "'='", "','", "'timeout'", "')'", "'type'", "'message'", "'end'", "'['", "']'", "'topic'", "'var'", "'event'", "'arg'", "'input'", "'->'", "'comment'", "'check'", "'=='", "'success'", "'failure'", "'running'", "'task'", "'in'", "'out'", "'par'", "'{'", "'}'", "'seq'", "'sel'", "'do'", "'if'", "'then'", "'mon'", "'exec'", "'timer'", "'chk'", "'invalid'", "'-'", "'.'"
    };
    public static final int T__50=50;
    public static final int RULE_BOOLEAN=7;
    public static final int T__19=19;
    public static final int T__15=15;
    public static final int T__16=16;
    public static final int T__17=17;
    public static final int T__18=18;
    public static final int T__55=55;
    public static final int T__12=12;
    public static final int T__13=13;
    public static final int T__14=14;
    public static final int T__51=51;
    public static final int T__52=52;
    public static final int T__53=53;
    public static final int T__54=54;
    public static final int RULE_ID=4;
    public static final int T__26=26;
    public static final int T__27=27;
    public static final int T__28=28;
    public static final int RULE_INT=5;
    public static final int T__29=29;
    public static final int T__22=22;
    public static final int RULE_ML_COMMENT=8;
    public static final int T__23=23;
    public static final int T__24=24;
    public static final int T__25=25;
    public static final int T__20=20;
    public static final int T__21=21;
    public static final int RULE_STRING=6;
    public static final int RULE_SL_COMMENT=9;
    public static final int T__37=37;
    public static final int T__38=38;
    public static final int T__39=39;
    public static final int T__33=33;
    public static final int T__34=34;
    public static final int T__35=35;
    public static final int T__36=36;
    public static final int EOF=-1;
    public static final int T__30=30;
    public static final int T__31=31;
    public static final int T__32=32;
    public static final int RULE_WS=10;
    public static final int RULE_ANY_OTHER=11;
    public static final int T__48=48;
    public static final int T__49=49;
    public static final int T__44=44;
    public static final int T__45=45;
    public static final int T__46=46;
    public static final int T__47=47;
    public static final int T__40=40;
    public static final int T__41=41;
    public static final int T__42=42;
    public static final int T__43=43;

    // delegates
    // delegators


        public InternalBTreeParser(TokenStream input) {
            this(input, new RecognizerSharedState());
        }
        public InternalBTreeParser(TokenStream input, RecognizerSharedState state) {
            super(input, state);
             
        }
        

    public String[] getTokenNames() { return InternalBTreeParser.tokenNames; }
    public String getGrammarFileName() { return "InternalBTree.g"; }



     	private BTreeGrammarAccess grammarAccess;

        public InternalBTreeParser(TokenStream input, BTreeGrammarAccess grammarAccess) {
            this(input);
            this.grammarAccess = grammarAccess;
            registerRules(grammarAccess.getGrammar());
        }

        @Override
        protected String getFirstRuleName() {
        	return "BehaviorModel";
       	}

       	@Override
       	protected BTreeGrammarAccess getGrammarAccess() {
       		return grammarAccess;
       	}




    // $ANTLR start "entryRuleBehaviorModel"
    // InternalBTree.g:64:1: entryRuleBehaviorModel returns [EObject current=null] : iv_ruleBehaviorModel= ruleBehaviorModel EOF ;
    public final EObject entryRuleBehaviorModel() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBehaviorModel = null;


        try {
            // InternalBTree.g:64:54: (iv_ruleBehaviorModel= ruleBehaviorModel EOF )
            // InternalBTree.g:65:2: iv_ruleBehaviorModel= ruleBehaviorModel EOF
            {
             newCompositeNode(grammarAccess.getBehaviorModelRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBehaviorModel=ruleBehaviorModel();

            state._fsp--;

             current =iv_ruleBehaviorModel; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBehaviorModel"


    // $ANTLR start "ruleBehaviorModel"
    // InternalBTree.g:71:1: ruleBehaviorModel returns [EObject current=null] : (otherlv_0= 'system' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ( (lv_simpleTypes_3_0= ruleSimpleType ) )* ( (lv_messageTypes_4_0= ruleMessageType ) )* ( (lv_rosTopics_5_0= ruleTopic ) )* ( (lv_bbVariables_6_0= ruleBBVar ) )* ( (lv_bbEvents_7_0= ruleBBEvent ) )* ( (lv_bbNodes_8_0= ruleBBNode ) )* ( (lv_checkNodes_9_0= ruleCheckNode ) )* ( (lv_taskNodes_10_0= ruleBehaviorNode ) )* otherlv_11= 'tree' otherlv_12= '(' otherlv_13= 'updatetime' otherlv_14= '=' ( (lv_updatetime_15_0= ruleFLOAT ) ) otherlv_16= ',' otherlv_17= 'timeout' otherlv_18= '=' ( (lv_timeout_19_0= ruleFLOAT ) ) otherlv_20= ')' ( (lv_tree_21_0= ruleBTree ) ) ) ;
    public final EObject ruleBehaviorModel() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_11=null;
        Token otherlv_12=null;
        Token otherlv_13=null;
        Token otherlv_14=null;
        Token otherlv_16=null;
        Token otherlv_17=null;
        Token otherlv_18=null;
        Token otherlv_20=null;
        EObject lv_simpleTypes_3_0 = null;

        EObject lv_messageTypes_4_0 = null;

        EObject lv_rosTopics_5_0 = null;

        EObject lv_bbVariables_6_0 = null;

        EObject lv_bbEvents_7_0 = null;

        EObject lv_bbNodes_8_0 = null;

        EObject lv_checkNodes_9_0 = null;

        EObject lv_taskNodes_10_0 = null;

        AntlrDatatypeRuleToken lv_updatetime_15_0 = null;

        AntlrDatatypeRuleToken lv_timeout_19_0 = null;

        EObject lv_tree_21_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:77:2: ( (otherlv_0= 'system' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ( (lv_simpleTypes_3_0= ruleSimpleType ) )* ( (lv_messageTypes_4_0= ruleMessageType ) )* ( (lv_rosTopics_5_0= ruleTopic ) )* ( (lv_bbVariables_6_0= ruleBBVar ) )* ( (lv_bbEvents_7_0= ruleBBEvent ) )* ( (lv_bbNodes_8_0= ruleBBNode ) )* ( (lv_checkNodes_9_0= ruleCheckNode ) )* ( (lv_taskNodes_10_0= ruleBehaviorNode ) )* otherlv_11= 'tree' otherlv_12= '(' otherlv_13= 'updatetime' otherlv_14= '=' ( (lv_updatetime_15_0= ruleFLOAT ) ) otherlv_16= ',' otherlv_17= 'timeout' otherlv_18= '=' ( (lv_timeout_19_0= ruleFLOAT ) ) otherlv_20= ')' ( (lv_tree_21_0= ruleBTree ) ) ) )
            // InternalBTree.g:78:2: (otherlv_0= 'system' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ( (lv_simpleTypes_3_0= ruleSimpleType ) )* ( (lv_messageTypes_4_0= ruleMessageType ) )* ( (lv_rosTopics_5_0= ruleTopic ) )* ( (lv_bbVariables_6_0= ruleBBVar ) )* ( (lv_bbEvents_7_0= ruleBBEvent ) )* ( (lv_bbNodes_8_0= ruleBBNode ) )* ( (lv_checkNodes_9_0= ruleCheckNode ) )* ( (lv_taskNodes_10_0= ruleBehaviorNode ) )* otherlv_11= 'tree' otherlv_12= '(' otherlv_13= 'updatetime' otherlv_14= '=' ( (lv_updatetime_15_0= ruleFLOAT ) ) otherlv_16= ',' otherlv_17= 'timeout' otherlv_18= '=' ( (lv_timeout_19_0= ruleFLOAT ) ) otherlv_20= ')' ( (lv_tree_21_0= ruleBTree ) ) )
            {
            // InternalBTree.g:78:2: (otherlv_0= 'system' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ( (lv_simpleTypes_3_0= ruleSimpleType ) )* ( (lv_messageTypes_4_0= ruleMessageType ) )* ( (lv_rosTopics_5_0= ruleTopic ) )* ( (lv_bbVariables_6_0= ruleBBVar ) )* ( (lv_bbEvents_7_0= ruleBBEvent ) )* ( (lv_bbNodes_8_0= ruleBBNode ) )* ( (lv_checkNodes_9_0= ruleCheckNode ) )* ( (lv_taskNodes_10_0= ruleBehaviorNode ) )* otherlv_11= 'tree' otherlv_12= '(' otherlv_13= 'updatetime' otherlv_14= '=' ( (lv_updatetime_15_0= ruleFLOAT ) ) otherlv_16= ',' otherlv_17= 'timeout' otherlv_18= '=' ( (lv_timeout_19_0= ruleFLOAT ) ) otherlv_20= ')' ( (lv_tree_21_0= ruleBTree ) ) )
            // InternalBTree.g:79:3: otherlv_0= 'system' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ( (lv_simpleTypes_3_0= ruleSimpleType ) )* ( (lv_messageTypes_4_0= ruleMessageType ) )* ( (lv_rosTopics_5_0= ruleTopic ) )* ( (lv_bbVariables_6_0= ruleBBVar ) )* ( (lv_bbEvents_7_0= ruleBBEvent ) )* ( (lv_bbNodes_8_0= ruleBBNode ) )* ( (lv_checkNodes_9_0= ruleCheckNode ) )* ( (lv_taskNodes_10_0= ruleBehaviorNode ) )* otherlv_11= 'tree' otherlv_12= '(' otherlv_13= 'updatetime' otherlv_14= '=' ( (lv_updatetime_15_0= ruleFLOAT ) ) otherlv_16= ',' otherlv_17= 'timeout' otherlv_18= '=' ( (lv_timeout_19_0= ruleFLOAT ) ) otherlv_20= ')' ( (lv_tree_21_0= ruleBTree ) )
            {
            otherlv_0=(Token)match(input,12,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getBehaviorModelAccess().getSystemKeyword_0());
            		
            // InternalBTree.g:83:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:84:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:84:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:85:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_4); 

            					newLeafNode(lv_name_1_0, grammarAccess.getBehaviorModelAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBehaviorModelRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            otherlv_2=(Token)match(input,13,FOLLOW_5); 

            			newLeafNode(otherlv_2, grammarAccess.getBehaviorModelAccess().getSemicolonKeyword_2());
            		
            // InternalBTree.g:105:3: ( (lv_simpleTypes_3_0= ruleSimpleType ) )*
            loop1:
            do {
                int alt1=2;
                int LA1_0 = input.LA(1);

                if ( (LA1_0==21) ) {
                    alt1=1;
                }


                switch (alt1) {
            	case 1 :
            	    // InternalBTree.g:106:4: (lv_simpleTypes_3_0= ruleSimpleType )
            	    {
            	    // InternalBTree.g:106:4: (lv_simpleTypes_3_0= ruleSimpleType )
            	    // InternalBTree.g:107:5: lv_simpleTypes_3_0= ruleSimpleType
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getSimpleTypesSimpleTypeParserRuleCall_3_0());
            	    				
            	    pushFollow(FOLLOW_5);
            	    lv_simpleTypes_3_0=ruleSimpleType();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"simpleTypes",
            	    						lv_simpleTypes_3_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.SimpleType");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop1;
                }
            } while (true);

            // InternalBTree.g:124:3: ( (lv_messageTypes_4_0= ruleMessageType ) )*
            loop2:
            do {
                int alt2=2;
                int LA2_0 = input.LA(1);

                if ( (LA2_0==22) ) {
                    alt2=1;
                }


                switch (alt2) {
            	case 1 :
            	    // InternalBTree.g:125:4: (lv_messageTypes_4_0= ruleMessageType )
            	    {
            	    // InternalBTree.g:125:4: (lv_messageTypes_4_0= ruleMessageType )
            	    // InternalBTree.g:126:5: lv_messageTypes_4_0= ruleMessageType
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getMessageTypesMessageTypeParserRuleCall_4_0());
            	    				
            	    pushFollow(FOLLOW_6);
            	    lv_messageTypes_4_0=ruleMessageType();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"messageTypes",
            	    						lv_messageTypes_4_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.MessageType");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop2;
                }
            } while (true);

            // InternalBTree.g:143:3: ( (lv_rosTopics_5_0= ruleTopic ) )*
            loop3:
            do {
                int alt3=2;
                int LA3_0 = input.LA(1);

                if ( (LA3_0==26) ) {
                    alt3=1;
                }


                switch (alt3) {
            	case 1 :
            	    // InternalBTree.g:144:4: (lv_rosTopics_5_0= ruleTopic )
            	    {
            	    // InternalBTree.g:144:4: (lv_rosTopics_5_0= ruleTopic )
            	    // InternalBTree.g:145:5: lv_rosTopics_5_0= ruleTopic
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getRosTopicsTopicParserRuleCall_5_0());
            	    				
            	    pushFollow(FOLLOW_7);
            	    lv_rosTopics_5_0=ruleTopic();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"rosTopics",
            	    						lv_rosTopics_5_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.Topic");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop3;
                }
            } while (true);

            // InternalBTree.g:162:3: ( (lv_bbVariables_6_0= ruleBBVar ) )*
            loop4:
            do {
                int alt4=2;
                int LA4_0 = input.LA(1);

                if ( (LA4_0==27) ) {
                    alt4=1;
                }


                switch (alt4) {
            	case 1 :
            	    // InternalBTree.g:163:4: (lv_bbVariables_6_0= ruleBBVar )
            	    {
            	    // InternalBTree.g:163:4: (lv_bbVariables_6_0= ruleBBVar )
            	    // InternalBTree.g:164:5: lv_bbVariables_6_0= ruleBBVar
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getBbVariablesBBVarParserRuleCall_6_0());
            	    				
            	    pushFollow(FOLLOW_8);
            	    lv_bbVariables_6_0=ruleBBVar();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"bbVariables",
            	    						lv_bbVariables_6_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.BBVar");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop4;
                }
            } while (true);

            // InternalBTree.g:181:3: ( (lv_bbEvents_7_0= ruleBBEvent ) )*
            loop5:
            do {
                int alt5=2;
                int LA5_0 = input.LA(1);

                if ( (LA5_0==28) ) {
                    alt5=1;
                }


                switch (alt5) {
            	case 1 :
            	    // InternalBTree.g:182:4: (lv_bbEvents_7_0= ruleBBEvent )
            	    {
            	    // InternalBTree.g:182:4: (lv_bbEvents_7_0= ruleBBEvent )
            	    // InternalBTree.g:183:5: lv_bbEvents_7_0= ruleBBEvent
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getBbEventsBBEventParserRuleCall_7_0());
            	    				
            	    pushFollow(FOLLOW_9);
            	    lv_bbEvents_7_0=ruleBBEvent();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"bbEvents",
            	    						lv_bbEvents_7_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.BBEvent");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop5;
                }
            } while (true);

            // InternalBTree.g:200:3: ( (lv_bbNodes_8_0= ruleBBNode ) )*
            loop6:
            do {
                int alt6=2;
                int LA6_0 = input.LA(1);

                if ( (LA6_0==30) ) {
                    alt6=1;
                }


                switch (alt6) {
            	case 1 :
            	    // InternalBTree.g:201:4: (lv_bbNodes_8_0= ruleBBNode )
            	    {
            	    // InternalBTree.g:201:4: (lv_bbNodes_8_0= ruleBBNode )
            	    // InternalBTree.g:202:5: lv_bbNodes_8_0= ruleBBNode
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getBbNodesBBNodeParserRuleCall_8_0());
            	    				
            	    pushFollow(FOLLOW_10);
            	    lv_bbNodes_8_0=ruleBBNode();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"bbNodes",
            	    						lv_bbNodes_8_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.BBNode");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop6;
                }
            } while (true);

            // InternalBTree.g:219:3: ( (lv_checkNodes_9_0= ruleCheckNode ) )*
            loop7:
            do {
                int alt7=2;
                int LA7_0 = input.LA(1);

                if ( (LA7_0==33) ) {
                    alt7=1;
                }


                switch (alt7) {
            	case 1 :
            	    // InternalBTree.g:220:4: (lv_checkNodes_9_0= ruleCheckNode )
            	    {
            	    // InternalBTree.g:220:4: (lv_checkNodes_9_0= ruleCheckNode )
            	    // InternalBTree.g:221:5: lv_checkNodes_9_0= ruleCheckNode
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getCheckNodesCheckNodeParserRuleCall_9_0());
            	    				
            	    pushFollow(FOLLOW_11);
            	    lv_checkNodes_9_0=ruleCheckNode();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"checkNodes",
            	    						lv_checkNodes_9_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.CheckNode");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop7;
                }
            } while (true);

            // InternalBTree.g:238:3: ( (lv_taskNodes_10_0= ruleBehaviorNode ) )*
            loop8:
            do {
                int alt8=2;
                int LA8_0 = input.LA(1);

                if ( ((LA8_0>=35 && LA8_0<=38)) ) {
                    alt8=1;
                }


                switch (alt8) {
            	case 1 :
            	    // InternalBTree.g:239:4: (lv_taskNodes_10_0= ruleBehaviorNode )
            	    {
            	    // InternalBTree.g:239:4: (lv_taskNodes_10_0= ruleBehaviorNode )
            	    // InternalBTree.g:240:5: lv_taskNodes_10_0= ruleBehaviorNode
            	    {

            	    					newCompositeNode(grammarAccess.getBehaviorModelAccess().getTaskNodesBehaviorNodeParserRuleCall_10_0());
            	    				
            	    pushFollow(FOLLOW_12);
            	    lv_taskNodes_10_0=ruleBehaviorNode();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            	    					}
            	    					add(
            	    						current,
            	    						"taskNodes",
            	    						lv_taskNodes_10_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.BehaviorNode");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop8;
                }
            } while (true);

            otherlv_11=(Token)match(input,14,FOLLOW_13); 

            			newLeafNode(otherlv_11, grammarAccess.getBehaviorModelAccess().getTreeKeyword_11());
            		
            otherlv_12=(Token)match(input,15,FOLLOW_14); 

            			newLeafNode(otherlv_12, grammarAccess.getBehaviorModelAccess().getLeftParenthesisKeyword_12());
            		
            otherlv_13=(Token)match(input,16,FOLLOW_15); 

            			newLeafNode(otherlv_13, grammarAccess.getBehaviorModelAccess().getUpdatetimeKeyword_13());
            		
            otherlv_14=(Token)match(input,17,FOLLOW_16); 

            			newLeafNode(otherlv_14, grammarAccess.getBehaviorModelAccess().getEqualsSignKeyword_14());
            		
            // InternalBTree.g:273:3: ( (lv_updatetime_15_0= ruleFLOAT ) )
            // InternalBTree.g:274:4: (lv_updatetime_15_0= ruleFLOAT )
            {
            // InternalBTree.g:274:4: (lv_updatetime_15_0= ruleFLOAT )
            // InternalBTree.g:275:5: lv_updatetime_15_0= ruleFLOAT
            {

            					newCompositeNode(grammarAccess.getBehaviorModelAccess().getUpdatetimeFLOATParserRuleCall_15_0());
            				
            pushFollow(FOLLOW_17);
            lv_updatetime_15_0=ruleFLOAT();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            					}
            					set(
            						current,
            						"updatetime",
            						lv_updatetime_15_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.FLOAT");
            					afterParserOrEnumRuleCall();
            				

            }


            }

            otherlv_16=(Token)match(input,18,FOLLOW_18); 

            			newLeafNode(otherlv_16, grammarAccess.getBehaviorModelAccess().getCommaKeyword_16());
            		
            otherlv_17=(Token)match(input,19,FOLLOW_15); 

            			newLeafNode(otherlv_17, grammarAccess.getBehaviorModelAccess().getTimeoutKeyword_17());
            		
            otherlv_18=(Token)match(input,17,FOLLOW_16); 

            			newLeafNode(otherlv_18, grammarAccess.getBehaviorModelAccess().getEqualsSignKeyword_18());
            		
            // InternalBTree.g:304:3: ( (lv_timeout_19_0= ruleFLOAT ) )
            // InternalBTree.g:305:4: (lv_timeout_19_0= ruleFLOAT )
            {
            // InternalBTree.g:305:4: (lv_timeout_19_0= ruleFLOAT )
            // InternalBTree.g:306:5: lv_timeout_19_0= ruleFLOAT
            {

            					newCompositeNode(grammarAccess.getBehaviorModelAccess().getTimeoutFLOATParserRuleCall_19_0());
            				
            pushFollow(FOLLOW_19);
            lv_timeout_19_0=ruleFLOAT();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            					}
            					set(
            						current,
            						"timeout",
            						lv_timeout_19_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.FLOAT");
            					afterParserOrEnumRuleCall();
            				

            }


            }

            otherlv_20=(Token)match(input,20,FOLLOW_20); 

            			newLeafNode(otherlv_20, grammarAccess.getBehaviorModelAccess().getRightParenthesisKeyword_20());
            		
            // InternalBTree.g:327:3: ( (lv_tree_21_0= ruleBTree ) )
            // InternalBTree.g:328:4: (lv_tree_21_0= ruleBTree )
            {
            // InternalBTree.g:328:4: (lv_tree_21_0= ruleBTree )
            // InternalBTree.g:329:5: lv_tree_21_0= ruleBTree
            {

            					newCompositeNode(grammarAccess.getBehaviorModelAccess().getTreeBTreeParserRuleCall_21_0());
            				
            pushFollow(FOLLOW_2);
            lv_tree_21_0=ruleBTree();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getBehaviorModelRule());
            					}
            					set(
            						current,
            						"tree",
            						lv_tree_21_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.BTree");
            					afterParserOrEnumRuleCall();
            				

            }


            }


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBehaviorModel"


    // $ANTLR start "entryRuleSimpleType"
    // InternalBTree.g:350:1: entryRuleSimpleType returns [EObject current=null] : iv_ruleSimpleType= ruleSimpleType EOF ;
    public final EObject entryRuleSimpleType() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleSimpleType = null;


        try {
            // InternalBTree.g:350:51: (iv_ruleSimpleType= ruleSimpleType EOF )
            // InternalBTree.g:351:2: iv_ruleSimpleType= ruleSimpleType EOF
            {
             newCompositeNode(grammarAccess.getSimpleTypeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleSimpleType=ruleSimpleType();

            state._fsp--;

             current =iv_ruleSimpleType; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleSimpleType"


    // $ANTLR start "ruleSimpleType"
    // InternalBTree.g:357:1: ruleSimpleType returns [EObject current=null] : (otherlv_0= 'type' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ) ;
    public final EObject ruleSimpleType() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;


        	enterRule();

        try {
            // InternalBTree.g:363:2: ( (otherlv_0= 'type' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ) )
            // InternalBTree.g:364:2: (otherlv_0= 'type' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' )
            {
            // InternalBTree.g:364:2: (otherlv_0= 'type' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' )
            // InternalBTree.g:365:3: otherlv_0= 'type' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';'
            {
            otherlv_0=(Token)match(input,21,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getSimpleTypeAccess().getTypeKeyword_0());
            		
            // InternalBTree.g:369:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:370:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:370:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:371:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_4); 

            					newLeafNode(lv_name_1_0, grammarAccess.getSimpleTypeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getSimpleTypeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            otherlv_2=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_2, grammarAccess.getSimpleTypeAccess().getSemicolonKeyword_2());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleSimpleType"


    // $ANTLR start "entryRuleMessageType"
    // InternalBTree.g:395:1: entryRuleMessageType returns [EObject current=null] : iv_ruleMessageType= ruleMessageType EOF ;
    public final EObject entryRuleMessageType() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleMessageType = null;


        try {
            // InternalBTree.g:395:52: (iv_ruleMessageType= ruleMessageType EOF )
            // InternalBTree.g:396:2: iv_ruleMessageType= ruleMessageType EOF
            {
             newCompositeNode(grammarAccess.getMessageTypeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleMessageType=ruleMessageType();

            state._fsp--;

             current =iv_ruleMessageType; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleMessageType"


    // $ANTLR start "ruleMessageType"
    // InternalBTree.g:402:1: ruleMessageType returns [EObject current=null] : (otherlv_0= 'message' ( (lv_name_1_0= RULE_ID ) ) ( (lv_package_2_0= RULE_ID ) ) ( (lv_fields_3_0= ruleField ) )* otherlv_4= 'end' (otherlv_5= ';' )? ) ;
    public final EObject ruleMessageType() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token lv_package_2_0=null;
        Token otherlv_4=null;
        Token otherlv_5=null;
        EObject lv_fields_3_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:408:2: ( (otherlv_0= 'message' ( (lv_name_1_0= RULE_ID ) ) ( (lv_package_2_0= RULE_ID ) ) ( (lv_fields_3_0= ruleField ) )* otherlv_4= 'end' (otherlv_5= ';' )? ) )
            // InternalBTree.g:409:2: (otherlv_0= 'message' ( (lv_name_1_0= RULE_ID ) ) ( (lv_package_2_0= RULE_ID ) ) ( (lv_fields_3_0= ruleField ) )* otherlv_4= 'end' (otherlv_5= ';' )? )
            {
            // InternalBTree.g:409:2: (otherlv_0= 'message' ( (lv_name_1_0= RULE_ID ) ) ( (lv_package_2_0= RULE_ID ) ) ( (lv_fields_3_0= ruleField ) )* otherlv_4= 'end' (otherlv_5= ';' )? )
            // InternalBTree.g:410:3: otherlv_0= 'message' ( (lv_name_1_0= RULE_ID ) ) ( (lv_package_2_0= RULE_ID ) ) ( (lv_fields_3_0= ruleField ) )* otherlv_4= 'end' (otherlv_5= ';' )?
            {
            otherlv_0=(Token)match(input,22,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getMessageTypeAccess().getMessageKeyword_0());
            		
            // InternalBTree.g:414:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:415:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:415:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:416:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_3); 

            					newLeafNode(lv_name_1_0, grammarAccess.getMessageTypeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getMessageTypeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:432:3: ( (lv_package_2_0= RULE_ID ) )
            // InternalBTree.g:433:4: (lv_package_2_0= RULE_ID )
            {
            // InternalBTree.g:433:4: (lv_package_2_0= RULE_ID )
            // InternalBTree.g:434:5: lv_package_2_0= RULE_ID
            {
            lv_package_2_0=(Token)match(input,RULE_ID,FOLLOW_21); 

            					newLeafNode(lv_package_2_0, grammarAccess.getMessageTypeAccess().getPackageIDTerminalRuleCall_2_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getMessageTypeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"package",
            						lv_package_2_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:450:3: ( (lv_fields_3_0= ruleField ) )*
            loop9:
            do {
                int alt9=2;
                int LA9_0 = input.LA(1);

                if ( (LA9_0==RULE_ID) ) {
                    alt9=1;
                }


                switch (alt9) {
            	case 1 :
            	    // InternalBTree.g:451:4: (lv_fields_3_0= ruleField )
            	    {
            	    // InternalBTree.g:451:4: (lv_fields_3_0= ruleField )
            	    // InternalBTree.g:452:5: lv_fields_3_0= ruleField
            	    {

            	    					newCompositeNode(grammarAccess.getMessageTypeAccess().getFieldsFieldParserRuleCall_3_0());
            	    				
            	    pushFollow(FOLLOW_21);
            	    lv_fields_3_0=ruleField();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getMessageTypeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"fields",
            	    						lv_fields_3_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.Field");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop9;
                }
            } while (true);

            otherlv_4=(Token)match(input,23,FOLLOW_22); 

            			newLeafNode(otherlv_4, grammarAccess.getMessageTypeAccess().getEndKeyword_4());
            		
            // InternalBTree.g:473:3: (otherlv_5= ';' )?
            int alt10=2;
            int LA10_0 = input.LA(1);

            if ( (LA10_0==13) ) {
                alt10=1;
            }
            switch (alt10) {
                case 1 :
                    // InternalBTree.g:474:4: otherlv_5= ';'
                    {
                    otherlv_5=(Token)match(input,13,FOLLOW_2); 

                    				newLeafNode(otherlv_5, grammarAccess.getMessageTypeAccess().getSemicolonKeyword_5());
                    			

                    }
                    break;

            }


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleMessageType"


    // $ANTLR start "entryRuleField"
    // InternalBTree.g:483:1: entryRuleField returns [EObject current=null] : iv_ruleField= ruleField EOF ;
    public final EObject entryRuleField() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleField = null;


        try {
            // InternalBTree.g:483:46: (iv_ruleField= ruleField EOF )
            // InternalBTree.g:484:2: iv_ruleField= ruleField EOF
            {
             newCompositeNode(grammarAccess.getFieldRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleField=ruleField();

            state._fsp--;

             current =iv_ruleField; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleField"


    // $ANTLR start "ruleField"
    // InternalBTree.g:490:1: ruleField returns [EObject current=null] : ( ( (otherlv_0= RULE_ID ) ) ( ( (lv_array_1_0= '[' ) ) ( (lv_count_2_0= RULE_INT ) )? otherlv_3= ']' )? ( (lv_name_4_0= RULE_ID ) ) otherlv_5= ';' ) ;
    public final EObject ruleField() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_array_1_0=null;
        Token lv_count_2_0=null;
        Token otherlv_3=null;
        Token lv_name_4_0=null;
        Token otherlv_5=null;


        	enterRule();

        try {
            // InternalBTree.g:496:2: ( ( ( (otherlv_0= RULE_ID ) ) ( ( (lv_array_1_0= '[' ) ) ( (lv_count_2_0= RULE_INT ) )? otherlv_3= ']' )? ( (lv_name_4_0= RULE_ID ) ) otherlv_5= ';' ) )
            // InternalBTree.g:497:2: ( ( (otherlv_0= RULE_ID ) ) ( ( (lv_array_1_0= '[' ) ) ( (lv_count_2_0= RULE_INT ) )? otherlv_3= ']' )? ( (lv_name_4_0= RULE_ID ) ) otherlv_5= ';' )
            {
            // InternalBTree.g:497:2: ( ( (otherlv_0= RULE_ID ) ) ( ( (lv_array_1_0= '[' ) ) ( (lv_count_2_0= RULE_INT ) )? otherlv_3= ']' )? ( (lv_name_4_0= RULE_ID ) ) otherlv_5= ';' )
            // InternalBTree.g:498:3: ( (otherlv_0= RULE_ID ) ) ( ( (lv_array_1_0= '[' ) ) ( (lv_count_2_0= RULE_INT ) )? otherlv_3= ']' )? ( (lv_name_4_0= RULE_ID ) ) otherlv_5= ';'
            {
            // InternalBTree.g:498:3: ( (otherlv_0= RULE_ID ) )
            // InternalBTree.g:499:4: (otherlv_0= RULE_ID )
            {
            // InternalBTree.g:499:4: (otherlv_0= RULE_ID )
            // InternalBTree.g:500:5: otherlv_0= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getFieldRule());
            					}
            				
            otherlv_0=(Token)match(input,RULE_ID,FOLLOW_23); 

            					newLeafNode(otherlv_0, grammarAccess.getFieldAccess().getTypeTypeCrossReference_0_0());
            				

            }


            }

            // InternalBTree.g:511:3: ( ( (lv_array_1_0= '[' ) ) ( (lv_count_2_0= RULE_INT ) )? otherlv_3= ']' )?
            int alt12=2;
            int LA12_0 = input.LA(1);

            if ( (LA12_0==24) ) {
                alt12=1;
            }
            switch (alt12) {
                case 1 :
                    // InternalBTree.g:512:4: ( (lv_array_1_0= '[' ) ) ( (lv_count_2_0= RULE_INT ) )? otherlv_3= ']'
                    {
                    // InternalBTree.g:512:4: ( (lv_array_1_0= '[' ) )
                    // InternalBTree.g:513:5: (lv_array_1_0= '[' )
                    {
                    // InternalBTree.g:513:5: (lv_array_1_0= '[' )
                    // InternalBTree.g:514:6: lv_array_1_0= '['
                    {
                    lv_array_1_0=(Token)match(input,24,FOLLOW_24); 

                    						newLeafNode(lv_array_1_0, grammarAccess.getFieldAccess().getArrayLeftSquareBracketKeyword_1_0_0());
                    					

                    						if (current==null) {
                    							current = createModelElement(grammarAccess.getFieldRule());
                    						}
                    						setWithLastConsumed(current, "array", lv_array_1_0, "[");
                    					

                    }


                    }

                    // InternalBTree.g:526:4: ( (lv_count_2_0= RULE_INT ) )?
                    int alt11=2;
                    int LA11_0 = input.LA(1);

                    if ( (LA11_0==RULE_INT) ) {
                        alt11=1;
                    }
                    switch (alt11) {
                        case 1 :
                            // InternalBTree.g:527:5: (lv_count_2_0= RULE_INT )
                            {
                            // InternalBTree.g:527:5: (lv_count_2_0= RULE_INT )
                            // InternalBTree.g:528:6: lv_count_2_0= RULE_INT
                            {
                            lv_count_2_0=(Token)match(input,RULE_INT,FOLLOW_25); 

                            						newLeafNode(lv_count_2_0, grammarAccess.getFieldAccess().getCountINTTerminalRuleCall_1_1_0());
                            					

                            						if (current==null) {
                            							current = createModelElement(grammarAccess.getFieldRule());
                            						}
                            						setWithLastConsumed(
                            							current,
                            							"count",
                            							lv_count_2_0,
                            							"org.eclipse.xtext.common.Terminals.INT");
                            					

                            }


                            }
                            break;

                    }

                    otherlv_3=(Token)match(input,25,FOLLOW_3); 

                    				newLeafNode(otherlv_3, grammarAccess.getFieldAccess().getRightSquareBracketKeyword_1_2());
                    			

                    }
                    break;

            }

            // InternalBTree.g:549:3: ( (lv_name_4_0= RULE_ID ) )
            // InternalBTree.g:550:4: (lv_name_4_0= RULE_ID )
            {
            // InternalBTree.g:550:4: (lv_name_4_0= RULE_ID )
            // InternalBTree.g:551:5: lv_name_4_0= RULE_ID
            {
            lv_name_4_0=(Token)match(input,RULE_ID,FOLLOW_4); 

            					newLeafNode(lv_name_4_0, grammarAccess.getFieldAccess().getNameIDTerminalRuleCall_2_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getFieldRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_4_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            otherlv_5=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_5, grammarAccess.getFieldAccess().getSemicolonKeyword_3());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleField"


    // $ANTLR start "entryRuleTopic"
    // InternalBTree.g:575:1: entryRuleTopic returns [EObject current=null] : iv_ruleTopic= ruleTopic EOF ;
    public final EObject entryRuleTopic() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleTopic = null;


        try {
            // InternalBTree.g:575:46: (iv_ruleTopic= ruleTopic EOF )
            // InternalBTree.g:576:2: iv_ruleTopic= ruleTopic EOF
            {
             newCompositeNode(grammarAccess.getTopicRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleTopic=ruleTopic();

            state._fsp--;

             current =iv_ruleTopic; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleTopic"


    // $ANTLR start "ruleTopic"
    // InternalBTree.g:582:1: ruleTopic returns [EObject current=null] : (otherlv_0= 'topic' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) ( (lv_topic_string_3_0= RULE_STRING ) ) otherlv_4= ';' ) ;
    public final EObject ruleTopic() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token otherlv_1=null;
        Token lv_name_2_0=null;
        Token lv_topic_string_3_0=null;
        Token otherlv_4=null;


        	enterRule();

        try {
            // InternalBTree.g:588:2: ( (otherlv_0= 'topic' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) ( (lv_topic_string_3_0= RULE_STRING ) ) otherlv_4= ';' ) )
            // InternalBTree.g:589:2: (otherlv_0= 'topic' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) ( (lv_topic_string_3_0= RULE_STRING ) ) otherlv_4= ';' )
            {
            // InternalBTree.g:589:2: (otherlv_0= 'topic' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) ( (lv_topic_string_3_0= RULE_STRING ) ) otherlv_4= ';' )
            // InternalBTree.g:590:3: otherlv_0= 'topic' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) ( (lv_topic_string_3_0= RULE_STRING ) ) otherlv_4= ';'
            {
            otherlv_0=(Token)match(input,26,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getTopicAccess().getTopicKeyword_0());
            		
            // InternalBTree.g:594:3: ( (otherlv_1= RULE_ID ) )
            // InternalBTree.g:595:4: (otherlv_1= RULE_ID )
            {
            // InternalBTree.g:595:4: (otherlv_1= RULE_ID )
            // InternalBTree.g:596:5: otherlv_1= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTopicRule());
            					}
            				
            otherlv_1=(Token)match(input,RULE_ID,FOLLOW_3); 

            					newLeafNode(otherlv_1, grammarAccess.getTopicAccess().getTypeMessageTypeCrossReference_1_0());
            				

            }


            }

            // InternalBTree.g:607:3: ( (lv_name_2_0= RULE_ID ) )
            // InternalBTree.g:608:4: (lv_name_2_0= RULE_ID )
            {
            // InternalBTree.g:608:4: (lv_name_2_0= RULE_ID )
            // InternalBTree.g:609:5: lv_name_2_0= RULE_ID
            {
            lv_name_2_0=(Token)match(input,RULE_ID,FOLLOW_26); 

            					newLeafNode(lv_name_2_0, grammarAccess.getTopicAccess().getNameIDTerminalRuleCall_2_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTopicRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_2_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:625:3: ( (lv_topic_string_3_0= RULE_STRING ) )
            // InternalBTree.g:626:4: (lv_topic_string_3_0= RULE_STRING )
            {
            // InternalBTree.g:626:4: (lv_topic_string_3_0= RULE_STRING )
            // InternalBTree.g:627:5: lv_topic_string_3_0= RULE_STRING
            {
            lv_topic_string_3_0=(Token)match(input,RULE_STRING,FOLLOW_4); 

            					newLeafNode(lv_topic_string_3_0, grammarAccess.getTopicAccess().getTopic_stringSTRINGTerminalRuleCall_3_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTopicRule());
            					}
            					setWithLastConsumed(
            						current,
            						"topic_string",
            						lv_topic_string_3_0,
            						"org.eclipse.xtext.common.Terminals.STRING");
            				

            }


            }

            otherlv_4=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_4, grammarAccess.getTopicAccess().getSemicolonKeyword_4());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleTopic"


    // $ANTLR start "entryRuleBBVar"
    // InternalBTree.g:651:1: entryRuleBBVar returns [EObject current=null] : iv_ruleBBVar= ruleBBVar EOF ;
    public final EObject entryRuleBBVar() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBBVar = null;


        try {
            // InternalBTree.g:651:46: (iv_ruleBBVar= ruleBBVar EOF )
            // InternalBTree.g:652:2: iv_ruleBBVar= ruleBBVar EOF
            {
             newCompositeNode(grammarAccess.getBBVarRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBBVar=ruleBBVar();

            state._fsp--;

             current =iv_ruleBBVar; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBBVar"


    // $ANTLR start "ruleBBVar"
    // InternalBTree.g:658:1: ruleBBVar returns [EObject current=null] : (otherlv_0= 'var' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) (otherlv_3= '=' ( (lv_default_4_0= ruleBASETYPE ) ) )? otherlv_5= ';' ) ;
    public final EObject ruleBBVar() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token otherlv_1=null;
        Token lv_name_2_0=null;
        Token otherlv_3=null;
        Token otherlv_5=null;
        AntlrDatatypeRuleToken lv_default_4_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:664:2: ( (otherlv_0= 'var' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) (otherlv_3= '=' ( (lv_default_4_0= ruleBASETYPE ) ) )? otherlv_5= ';' ) )
            // InternalBTree.g:665:2: (otherlv_0= 'var' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) (otherlv_3= '=' ( (lv_default_4_0= ruleBASETYPE ) ) )? otherlv_5= ';' )
            {
            // InternalBTree.g:665:2: (otherlv_0= 'var' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) (otherlv_3= '=' ( (lv_default_4_0= ruleBASETYPE ) ) )? otherlv_5= ';' )
            // InternalBTree.g:666:3: otherlv_0= 'var' ( (otherlv_1= RULE_ID ) ) ( (lv_name_2_0= RULE_ID ) ) (otherlv_3= '=' ( (lv_default_4_0= ruleBASETYPE ) ) )? otherlv_5= ';'
            {
            otherlv_0=(Token)match(input,27,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getBBVarAccess().getVarKeyword_0());
            		
            // InternalBTree.g:670:3: ( (otherlv_1= RULE_ID ) )
            // InternalBTree.g:671:4: (otherlv_1= RULE_ID )
            {
            // InternalBTree.g:671:4: (otherlv_1= RULE_ID )
            // InternalBTree.g:672:5: otherlv_1= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBBVarRule());
            					}
            				
            otherlv_1=(Token)match(input,RULE_ID,FOLLOW_3); 

            					newLeafNode(otherlv_1, grammarAccess.getBBVarAccess().getTypeTypeCrossReference_1_0());
            				

            }


            }

            // InternalBTree.g:683:3: ( (lv_name_2_0= RULE_ID ) )
            // InternalBTree.g:684:4: (lv_name_2_0= RULE_ID )
            {
            // InternalBTree.g:684:4: (lv_name_2_0= RULE_ID )
            // InternalBTree.g:685:5: lv_name_2_0= RULE_ID
            {
            lv_name_2_0=(Token)match(input,RULE_ID,FOLLOW_27); 

            					newLeafNode(lv_name_2_0, grammarAccess.getBBVarAccess().getNameIDTerminalRuleCall_2_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBBVarRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_2_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:701:3: (otherlv_3= '=' ( (lv_default_4_0= ruleBASETYPE ) ) )?
            int alt13=2;
            int LA13_0 = input.LA(1);

            if ( (LA13_0==17) ) {
                alt13=1;
            }
            switch (alt13) {
                case 1 :
                    // InternalBTree.g:702:4: otherlv_3= '=' ( (lv_default_4_0= ruleBASETYPE ) )
                    {
                    otherlv_3=(Token)match(input,17,FOLLOW_28); 

                    				newLeafNode(otherlv_3, grammarAccess.getBBVarAccess().getEqualsSignKeyword_3_0());
                    			
                    // InternalBTree.g:706:4: ( (lv_default_4_0= ruleBASETYPE ) )
                    // InternalBTree.g:707:5: (lv_default_4_0= ruleBASETYPE )
                    {
                    // InternalBTree.g:707:5: (lv_default_4_0= ruleBASETYPE )
                    // InternalBTree.g:708:6: lv_default_4_0= ruleBASETYPE
                    {

                    						newCompositeNode(grammarAccess.getBBVarAccess().getDefaultBASETYPEParserRuleCall_3_1_0());
                    					
                    pushFollow(FOLLOW_4);
                    lv_default_4_0=ruleBASETYPE();

                    state._fsp--;


                    						if (current==null) {
                    							current = createModelElementForParent(grammarAccess.getBBVarRule());
                    						}
                    						set(
                    							current,
                    							"default",
                    							lv_default_4_0,
                    							"edu.vanderbilt.isis.alc.btree.BTree.BASETYPE");
                    						afterParserOrEnumRuleCall();
                    					

                    }


                    }


                    }
                    break;

            }

            otherlv_5=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_5, grammarAccess.getBBVarAccess().getSemicolonKeyword_4());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBBVar"


    // $ANTLR start "entryRuleBBEvent"
    // InternalBTree.g:734:1: entryRuleBBEvent returns [EObject current=null] : iv_ruleBBEvent= ruleBBEvent EOF ;
    public final EObject entryRuleBBEvent() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBBEvent = null;


        try {
            // InternalBTree.g:734:48: (iv_ruleBBEvent= ruleBBEvent EOF )
            // InternalBTree.g:735:2: iv_ruleBBEvent= ruleBBEvent EOF
            {
             newCompositeNode(grammarAccess.getBBEventRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBBEvent=ruleBBEvent();

            state._fsp--;

             current =iv_ruleBBEvent; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBBEvent"


    // $ANTLR start "ruleBBEvent"
    // InternalBTree.g:741:1: ruleBBEvent returns [EObject current=null] : (otherlv_0= 'event' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= ';' ) ;
    public final EObject ruleBBEvent() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_3=null;


        	enterRule();

        try {
            // InternalBTree.g:747:2: ( (otherlv_0= 'event' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= ';' ) )
            // InternalBTree.g:748:2: (otherlv_0= 'event' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= ';' )
            {
            // InternalBTree.g:748:2: (otherlv_0= 'event' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= ';' )
            // InternalBTree.g:749:3: otherlv_0= 'event' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= ';'
            {
            otherlv_0=(Token)match(input,28,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getBBEventAccess().getEventKeyword_0());
            		
            // InternalBTree.g:753:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:754:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:754:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:755:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_3); 

            					newLeafNode(lv_name_1_0, grammarAccess.getBBEventAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBBEventRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:771:3: ( (otherlv_2= RULE_ID ) )
            // InternalBTree.g:772:4: (otherlv_2= RULE_ID )
            {
            // InternalBTree.g:772:4: (otherlv_2= RULE_ID )
            // InternalBTree.g:773:5: otherlv_2= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBBEventRule());
            					}
            				
            otherlv_2=(Token)match(input,RULE_ID,FOLLOW_4); 

            					newLeafNode(otherlv_2, grammarAccess.getBBEventAccess().getTopicTopicCrossReference_2_0());
            				

            }


            }

            otherlv_3=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_3, grammarAccess.getBBEventAccess().getSemicolonKeyword_3());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBBEvent"


    // $ANTLR start "entryRuleArg"
    // InternalBTree.g:792:1: entryRuleArg returns [EObject current=null] : iv_ruleArg= ruleArg EOF ;
    public final EObject entryRuleArg() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleArg = null;


        try {
            // InternalBTree.g:792:44: (iv_ruleArg= ruleArg EOF )
            // InternalBTree.g:793:2: iv_ruleArg= ruleArg EOF
            {
             newCompositeNode(grammarAccess.getArgRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleArg=ruleArg();

            state._fsp--;

             current =iv_ruleArg; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleArg"


    // $ANTLR start "ruleArg"
    // InternalBTree.g:799:1: ruleArg returns [EObject current=null] : (otherlv_0= 'arg' ( (otherlv_1= RULE_ID ) ) ( ( (lv_array_2_0= '[' ) ) ( (lv_count_3_0= RULE_INT ) )? otherlv_4= ']' )? ( (lv_name_5_0= RULE_ID ) ) (otherlv_6= '=' ( (lv_default_7_0= ruleDefaultType ) ) )? otherlv_8= ';' ) ;
    public final EObject ruleArg() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token otherlv_1=null;
        Token lv_array_2_0=null;
        Token lv_count_3_0=null;
        Token otherlv_4=null;
        Token lv_name_5_0=null;
        Token otherlv_6=null;
        Token otherlv_8=null;
        EObject lv_default_7_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:805:2: ( (otherlv_0= 'arg' ( (otherlv_1= RULE_ID ) ) ( ( (lv_array_2_0= '[' ) ) ( (lv_count_3_0= RULE_INT ) )? otherlv_4= ']' )? ( (lv_name_5_0= RULE_ID ) ) (otherlv_6= '=' ( (lv_default_7_0= ruleDefaultType ) ) )? otherlv_8= ';' ) )
            // InternalBTree.g:806:2: (otherlv_0= 'arg' ( (otherlv_1= RULE_ID ) ) ( ( (lv_array_2_0= '[' ) ) ( (lv_count_3_0= RULE_INT ) )? otherlv_4= ']' )? ( (lv_name_5_0= RULE_ID ) ) (otherlv_6= '=' ( (lv_default_7_0= ruleDefaultType ) ) )? otherlv_8= ';' )
            {
            // InternalBTree.g:806:2: (otherlv_0= 'arg' ( (otherlv_1= RULE_ID ) ) ( ( (lv_array_2_0= '[' ) ) ( (lv_count_3_0= RULE_INT ) )? otherlv_4= ']' )? ( (lv_name_5_0= RULE_ID ) ) (otherlv_6= '=' ( (lv_default_7_0= ruleDefaultType ) ) )? otherlv_8= ';' )
            // InternalBTree.g:807:3: otherlv_0= 'arg' ( (otherlv_1= RULE_ID ) ) ( ( (lv_array_2_0= '[' ) ) ( (lv_count_3_0= RULE_INT ) )? otherlv_4= ']' )? ( (lv_name_5_0= RULE_ID ) ) (otherlv_6= '=' ( (lv_default_7_0= ruleDefaultType ) ) )? otherlv_8= ';'
            {
            otherlv_0=(Token)match(input,29,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getArgAccess().getArgKeyword_0());
            		
            // InternalBTree.g:811:3: ( (otherlv_1= RULE_ID ) )
            // InternalBTree.g:812:4: (otherlv_1= RULE_ID )
            {
            // InternalBTree.g:812:4: (otherlv_1= RULE_ID )
            // InternalBTree.g:813:5: otherlv_1= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getArgRule());
            					}
            				
            otherlv_1=(Token)match(input,RULE_ID,FOLLOW_23); 

            					newLeafNode(otherlv_1, grammarAccess.getArgAccess().getTypeTypeCrossReference_1_0());
            				

            }


            }

            // InternalBTree.g:824:3: ( ( (lv_array_2_0= '[' ) ) ( (lv_count_3_0= RULE_INT ) )? otherlv_4= ']' )?
            int alt15=2;
            int LA15_0 = input.LA(1);

            if ( (LA15_0==24) ) {
                alt15=1;
            }
            switch (alt15) {
                case 1 :
                    // InternalBTree.g:825:4: ( (lv_array_2_0= '[' ) ) ( (lv_count_3_0= RULE_INT ) )? otherlv_4= ']'
                    {
                    // InternalBTree.g:825:4: ( (lv_array_2_0= '[' ) )
                    // InternalBTree.g:826:5: (lv_array_2_0= '[' )
                    {
                    // InternalBTree.g:826:5: (lv_array_2_0= '[' )
                    // InternalBTree.g:827:6: lv_array_2_0= '['
                    {
                    lv_array_2_0=(Token)match(input,24,FOLLOW_24); 

                    						newLeafNode(lv_array_2_0, grammarAccess.getArgAccess().getArrayLeftSquareBracketKeyword_2_0_0());
                    					

                    						if (current==null) {
                    							current = createModelElement(grammarAccess.getArgRule());
                    						}
                    						setWithLastConsumed(current, "array", lv_array_2_0, "[");
                    					

                    }


                    }

                    // InternalBTree.g:839:4: ( (lv_count_3_0= RULE_INT ) )?
                    int alt14=2;
                    int LA14_0 = input.LA(1);

                    if ( (LA14_0==RULE_INT) ) {
                        alt14=1;
                    }
                    switch (alt14) {
                        case 1 :
                            // InternalBTree.g:840:5: (lv_count_3_0= RULE_INT )
                            {
                            // InternalBTree.g:840:5: (lv_count_3_0= RULE_INT )
                            // InternalBTree.g:841:6: lv_count_3_0= RULE_INT
                            {
                            lv_count_3_0=(Token)match(input,RULE_INT,FOLLOW_25); 

                            						newLeafNode(lv_count_3_0, grammarAccess.getArgAccess().getCountINTTerminalRuleCall_2_1_0());
                            					

                            						if (current==null) {
                            							current = createModelElement(grammarAccess.getArgRule());
                            						}
                            						setWithLastConsumed(
                            							current,
                            							"count",
                            							lv_count_3_0,
                            							"org.eclipse.xtext.common.Terminals.INT");
                            					

                            }


                            }
                            break;

                    }

                    otherlv_4=(Token)match(input,25,FOLLOW_3); 

                    				newLeafNode(otherlv_4, grammarAccess.getArgAccess().getRightSquareBracketKeyword_2_2());
                    			

                    }
                    break;

            }

            // InternalBTree.g:862:3: ( (lv_name_5_0= RULE_ID ) )
            // InternalBTree.g:863:4: (lv_name_5_0= RULE_ID )
            {
            // InternalBTree.g:863:4: (lv_name_5_0= RULE_ID )
            // InternalBTree.g:864:5: lv_name_5_0= RULE_ID
            {
            lv_name_5_0=(Token)match(input,RULE_ID,FOLLOW_27); 

            					newLeafNode(lv_name_5_0, grammarAccess.getArgAccess().getNameIDTerminalRuleCall_3_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getArgRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_5_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:880:3: (otherlv_6= '=' ( (lv_default_7_0= ruleDefaultType ) ) )?
            int alt16=2;
            int LA16_0 = input.LA(1);

            if ( (LA16_0==17) ) {
                alt16=1;
            }
            switch (alt16) {
                case 1 :
                    // InternalBTree.g:881:4: otherlv_6= '=' ( (lv_default_7_0= ruleDefaultType ) )
                    {
                    otherlv_6=(Token)match(input,17,FOLLOW_29); 

                    				newLeafNode(otherlv_6, grammarAccess.getArgAccess().getEqualsSignKeyword_4_0());
                    			
                    // InternalBTree.g:885:4: ( (lv_default_7_0= ruleDefaultType ) )
                    // InternalBTree.g:886:5: (lv_default_7_0= ruleDefaultType )
                    {
                    // InternalBTree.g:886:5: (lv_default_7_0= ruleDefaultType )
                    // InternalBTree.g:887:6: lv_default_7_0= ruleDefaultType
                    {

                    						newCompositeNode(grammarAccess.getArgAccess().getDefaultDefaultTypeParserRuleCall_4_1_0());
                    					
                    pushFollow(FOLLOW_4);
                    lv_default_7_0=ruleDefaultType();

                    state._fsp--;


                    						if (current==null) {
                    							current = createModelElementForParent(grammarAccess.getArgRule());
                    						}
                    						set(
                    							current,
                    							"default",
                    							lv_default_7_0,
                    							"edu.vanderbilt.isis.alc.btree.BTree.DefaultType");
                    						afterParserOrEnumRuleCall();
                    					

                    }


                    }


                    }
                    break;

            }

            otherlv_8=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_8, grammarAccess.getArgAccess().getSemicolonKeyword_5());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleArg"


    // $ANTLR start "entryRuleDefaultType"
    // InternalBTree.g:913:1: entryRuleDefaultType returns [EObject current=null] : iv_ruleDefaultType= ruleDefaultType EOF ;
    public final EObject entryRuleDefaultType() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleDefaultType = null;


        try {
            // InternalBTree.g:913:52: (iv_ruleDefaultType= ruleDefaultType EOF )
            // InternalBTree.g:914:2: iv_ruleDefaultType= ruleDefaultType EOF
            {
             newCompositeNode(grammarAccess.getDefaultTypeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleDefaultType=ruleDefaultType();

            state._fsp--;

             current =iv_ruleDefaultType; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleDefaultType"


    // $ANTLR start "ruleDefaultType"
    // InternalBTree.g:920:1: ruleDefaultType returns [EObject current=null] : ( ( () ruleBASETYPE ) | this_BaseArrayType_2= ruleBaseArrayType ) ;
    public final EObject ruleDefaultType() throws RecognitionException {
        EObject current = null;

        EObject this_BaseArrayType_2 = null;



        	enterRule();

        try {
            // InternalBTree.g:926:2: ( ( ( () ruleBASETYPE ) | this_BaseArrayType_2= ruleBaseArrayType ) )
            // InternalBTree.g:927:2: ( ( () ruleBASETYPE ) | this_BaseArrayType_2= ruleBaseArrayType )
            {
            // InternalBTree.g:927:2: ( ( () ruleBASETYPE ) | this_BaseArrayType_2= ruleBaseArrayType )
            int alt17=2;
            int LA17_0 = input.LA(1);

            if ( ((LA17_0>=RULE_INT && LA17_0<=RULE_BOOLEAN)||LA17_0==54) ) {
                alt17=1;
            }
            else if ( (LA17_0==24) ) {
                alt17=2;
            }
            else {
                NoViableAltException nvae =
                    new NoViableAltException("", 17, 0, input);

                throw nvae;
            }
            switch (alt17) {
                case 1 :
                    // InternalBTree.g:928:3: ( () ruleBASETYPE )
                    {
                    // InternalBTree.g:928:3: ( () ruleBASETYPE )
                    // InternalBTree.g:929:4: () ruleBASETYPE
                    {
                    // InternalBTree.g:929:4: ()
                    // InternalBTree.g:930:5: 
                    {

                    					current = forceCreateModelElement(
                    						grammarAccess.getDefaultTypeAccess().getDefaultTypeAction_0_0(),
                    						current);
                    				

                    }


                    				newCompositeNode(grammarAccess.getDefaultTypeAccess().getBASETYPEParserRuleCall_0_1());
                    			
                    pushFollow(FOLLOW_2);
                    ruleBASETYPE();

                    state._fsp--;


                    				afterParserOrEnumRuleCall();
                    			

                    }


                    }
                    break;
                case 2 :
                    // InternalBTree.g:945:3: this_BaseArrayType_2= ruleBaseArrayType
                    {

                    			newCompositeNode(grammarAccess.getDefaultTypeAccess().getBaseArrayTypeParserRuleCall_1());
                    		
                    pushFollow(FOLLOW_2);
                    this_BaseArrayType_2=ruleBaseArrayType();

                    state._fsp--;


                    			current = this_BaseArrayType_2;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleDefaultType"


    // $ANTLR start "entryRuleBaseArrayType"
    // InternalBTree.g:957:1: entryRuleBaseArrayType returns [EObject current=null] : iv_ruleBaseArrayType= ruleBaseArrayType EOF ;
    public final EObject entryRuleBaseArrayType() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBaseArrayType = null;


        try {
            // InternalBTree.g:957:54: (iv_ruleBaseArrayType= ruleBaseArrayType EOF )
            // InternalBTree.g:958:2: iv_ruleBaseArrayType= ruleBaseArrayType EOF
            {
             newCompositeNode(grammarAccess.getBaseArrayTypeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBaseArrayType=ruleBaseArrayType();

            state._fsp--;

             current =iv_ruleBaseArrayType; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBaseArrayType"


    // $ANTLR start "ruleBaseArrayType"
    // InternalBTree.g:964:1: ruleBaseArrayType returns [EObject current=null] : (otherlv_0= '[' ( (lv_values_1_0= ruleBASETYPE ) ) (otherlv_2= ',' ( (lv_values_3_0= ruleBASETYPE ) ) )* otherlv_4= ']' ) ;
    public final EObject ruleBaseArrayType() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token otherlv_2=null;
        Token otherlv_4=null;
        AntlrDatatypeRuleToken lv_values_1_0 = null;

        AntlrDatatypeRuleToken lv_values_3_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:970:2: ( (otherlv_0= '[' ( (lv_values_1_0= ruleBASETYPE ) ) (otherlv_2= ',' ( (lv_values_3_0= ruleBASETYPE ) ) )* otherlv_4= ']' ) )
            // InternalBTree.g:971:2: (otherlv_0= '[' ( (lv_values_1_0= ruleBASETYPE ) ) (otherlv_2= ',' ( (lv_values_3_0= ruleBASETYPE ) ) )* otherlv_4= ']' )
            {
            // InternalBTree.g:971:2: (otherlv_0= '[' ( (lv_values_1_0= ruleBASETYPE ) ) (otherlv_2= ',' ( (lv_values_3_0= ruleBASETYPE ) ) )* otherlv_4= ']' )
            // InternalBTree.g:972:3: otherlv_0= '[' ( (lv_values_1_0= ruleBASETYPE ) ) (otherlv_2= ',' ( (lv_values_3_0= ruleBASETYPE ) ) )* otherlv_4= ']'
            {
            otherlv_0=(Token)match(input,24,FOLLOW_28); 

            			newLeafNode(otherlv_0, grammarAccess.getBaseArrayTypeAccess().getLeftSquareBracketKeyword_0());
            		
            // InternalBTree.g:976:3: ( (lv_values_1_0= ruleBASETYPE ) )
            // InternalBTree.g:977:4: (lv_values_1_0= ruleBASETYPE )
            {
            // InternalBTree.g:977:4: (lv_values_1_0= ruleBASETYPE )
            // InternalBTree.g:978:5: lv_values_1_0= ruleBASETYPE
            {

            					newCompositeNode(grammarAccess.getBaseArrayTypeAccess().getValuesBASETYPEParserRuleCall_1_0());
            				
            pushFollow(FOLLOW_30);
            lv_values_1_0=ruleBASETYPE();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getBaseArrayTypeRule());
            					}
            					add(
            						current,
            						"values",
            						lv_values_1_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.BASETYPE");
            					afterParserOrEnumRuleCall();
            				

            }


            }

            // InternalBTree.g:995:3: (otherlv_2= ',' ( (lv_values_3_0= ruleBASETYPE ) ) )*
            loop18:
            do {
                int alt18=2;
                int LA18_0 = input.LA(1);

                if ( (LA18_0==18) ) {
                    alt18=1;
                }


                switch (alt18) {
            	case 1 :
            	    // InternalBTree.g:996:4: otherlv_2= ',' ( (lv_values_3_0= ruleBASETYPE ) )
            	    {
            	    otherlv_2=(Token)match(input,18,FOLLOW_28); 

            	    				newLeafNode(otherlv_2, grammarAccess.getBaseArrayTypeAccess().getCommaKeyword_2_0());
            	    			
            	    // InternalBTree.g:1000:4: ( (lv_values_3_0= ruleBASETYPE ) )
            	    // InternalBTree.g:1001:5: (lv_values_3_0= ruleBASETYPE )
            	    {
            	    // InternalBTree.g:1001:5: (lv_values_3_0= ruleBASETYPE )
            	    // InternalBTree.g:1002:6: lv_values_3_0= ruleBASETYPE
            	    {

            	    						newCompositeNode(grammarAccess.getBaseArrayTypeAccess().getValuesBASETYPEParserRuleCall_2_1_0());
            	    					
            	    pushFollow(FOLLOW_30);
            	    lv_values_3_0=ruleBASETYPE();

            	    state._fsp--;


            	    						if (current==null) {
            	    							current = createModelElementForParent(grammarAccess.getBaseArrayTypeRule());
            	    						}
            	    						add(
            	    							current,
            	    							"values",
            	    							lv_values_3_0,
            	    							"edu.vanderbilt.isis.alc.btree.BTree.BASETYPE");
            	    						afterParserOrEnumRuleCall();
            	    					

            	    }


            	    }


            	    }
            	    break;

            	default :
            	    break loop18;
                }
            } while (true);

            otherlv_4=(Token)match(input,25,FOLLOW_2); 

            			newLeafNode(otherlv_4, grammarAccess.getBaseArrayTypeAccess().getRightSquareBracketKeyword_3());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBaseArrayType"


    // $ANTLR start "entryRuleBBNode"
    // InternalBTree.g:1028:1: entryRuleBBNode returns [EObject current=null] : iv_ruleBBNode= ruleBBNode EOF ;
    public final EObject entryRuleBBNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBBNode = null;


        try {
            // InternalBTree.g:1028:47: (iv_ruleBBNode= ruleBBNode EOF )
            // InternalBTree.g:1029:2: iv_ruleBBNode= ruleBBNode EOF
            {
             newCompositeNode(grammarAccess.getBBNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBBNode=ruleBBNode();

            state._fsp--;

             current =iv_ruleBBNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBBNode"


    // $ANTLR start "ruleBBNode"
    // InternalBTree.g:1035:1: ruleBBNode returns [EObject current=null] : (otherlv_0= 'input' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= '->' ( (otherlv_4= RULE_ID ) ) ( (lv_bb_vars_5_0= ruleBBVar ) )* ( (lv_args_6_0= ruleArg ) )* (otherlv_7= 'comment' ( (lv_comment_8_0= RULE_STRING ) ) )? otherlv_9= 'end' (otherlv_10= ';' )? ) ;
    public final EObject ruleBBNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_3=null;
        Token otherlv_4=null;
        Token otherlv_7=null;
        Token lv_comment_8_0=null;
        Token otherlv_9=null;
        Token otherlv_10=null;
        EObject lv_bb_vars_5_0 = null;

        EObject lv_args_6_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1041:2: ( (otherlv_0= 'input' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= '->' ( (otherlv_4= RULE_ID ) ) ( (lv_bb_vars_5_0= ruleBBVar ) )* ( (lv_args_6_0= ruleArg ) )* (otherlv_7= 'comment' ( (lv_comment_8_0= RULE_STRING ) ) )? otherlv_9= 'end' (otherlv_10= ';' )? ) )
            // InternalBTree.g:1042:2: (otherlv_0= 'input' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= '->' ( (otherlv_4= RULE_ID ) ) ( (lv_bb_vars_5_0= ruleBBVar ) )* ( (lv_args_6_0= ruleArg ) )* (otherlv_7= 'comment' ( (lv_comment_8_0= RULE_STRING ) ) )? otherlv_9= 'end' (otherlv_10= ';' )? )
            {
            // InternalBTree.g:1042:2: (otherlv_0= 'input' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= '->' ( (otherlv_4= RULE_ID ) ) ( (lv_bb_vars_5_0= ruleBBVar ) )* ( (lv_args_6_0= ruleArg ) )* (otherlv_7= 'comment' ( (lv_comment_8_0= RULE_STRING ) ) )? otherlv_9= 'end' (otherlv_10= ';' )? )
            // InternalBTree.g:1043:3: otherlv_0= 'input' ( (lv_name_1_0= RULE_ID ) ) ( (otherlv_2= RULE_ID ) ) otherlv_3= '->' ( (otherlv_4= RULE_ID ) ) ( (lv_bb_vars_5_0= ruleBBVar ) )* ( (lv_args_6_0= ruleArg ) )* (otherlv_7= 'comment' ( (lv_comment_8_0= RULE_STRING ) ) )? otherlv_9= 'end' (otherlv_10= ';' )?
            {
            otherlv_0=(Token)match(input,30,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getBBNodeAccess().getInputKeyword_0());
            		
            // InternalBTree.g:1047:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:1048:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:1048:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:1049:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_3); 

            					newLeafNode(lv_name_1_0, grammarAccess.getBBNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBBNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:1065:3: ( (otherlv_2= RULE_ID ) )
            // InternalBTree.g:1066:4: (otherlv_2= RULE_ID )
            {
            // InternalBTree.g:1066:4: (otherlv_2= RULE_ID )
            // InternalBTree.g:1067:5: otherlv_2= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBBNodeRule());
            					}
            				
            otherlv_2=(Token)match(input,RULE_ID,FOLLOW_31); 

            					newLeafNode(otherlv_2, grammarAccess.getBBNodeAccess().getInput_topicTopicCrossReference_2_0());
            				

            }


            }

            otherlv_3=(Token)match(input,31,FOLLOW_3); 

            			newLeafNode(otherlv_3, grammarAccess.getBBNodeAccess().getHyphenMinusGreaterThanSignKeyword_3());
            		
            // InternalBTree.g:1082:3: ( (otherlv_4= RULE_ID ) )
            // InternalBTree.g:1083:4: (otherlv_4= RULE_ID )
            {
            // InternalBTree.g:1083:4: (otherlv_4= RULE_ID )
            // InternalBTree.g:1084:5: otherlv_4= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getBBNodeRule());
            					}
            				
            otherlv_4=(Token)match(input,RULE_ID,FOLLOW_32); 

            					newLeafNode(otherlv_4, grammarAccess.getBBNodeAccess().getTopic_bbvarBBVarCrossReference_4_0());
            				

            }


            }

            // InternalBTree.g:1095:3: ( (lv_bb_vars_5_0= ruleBBVar ) )*
            loop19:
            do {
                int alt19=2;
                int LA19_0 = input.LA(1);

                if ( (LA19_0==27) ) {
                    alt19=1;
                }


                switch (alt19) {
            	case 1 :
            	    // InternalBTree.g:1096:4: (lv_bb_vars_5_0= ruleBBVar )
            	    {
            	    // InternalBTree.g:1096:4: (lv_bb_vars_5_0= ruleBBVar )
            	    // InternalBTree.g:1097:5: lv_bb_vars_5_0= ruleBBVar
            	    {

            	    					newCompositeNode(grammarAccess.getBBNodeAccess().getBb_varsBBVarParserRuleCall_5_0());
            	    				
            	    pushFollow(FOLLOW_32);
            	    lv_bb_vars_5_0=ruleBBVar();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBBNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"bb_vars",
            	    						lv_bb_vars_5_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.BBVar");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop19;
                }
            } while (true);

            // InternalBTree.g:1114:3: ( (lv_args_6_0= ruleArg ) )*
            loop20:
            do {
                int alt20=2;
                int LA20_0 = input.LA(1);

                if ( (LA20_0==29) ) {
                    alt20=1;
                }


                switch (alt20) {
            	case 1 :
            	    // InternalBTree.g:1115:4: (lv_args_6_0= ruleArg )
            	    {
            	    // InternalBTree.g:1115:4: (lv_args_6_0= ruleArg )
            	    // InternalBTree.g:1116:5: lv_args_6_0= ruleArg
            	    {

            	    					newCompositeNode(grammarAccess.getBBNodeAccess().getArgsArgParserRuleCall_6_0());
            	    				
            	    pushFollow(FOLLOW_33);
            	    lv_args_6_0=ruleArg();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getBBNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"args",
            	    						lv_args_6_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.Arg");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop20;
                }
            } while (true);

            // InternalBTree.g:1133:3: (otherlv_7= 'comment' ( (lv_comment_8_0= RULE_STRING ) ) )?
            int alt21=2;
            int LA21_0 = input.LA(1);

            if ( (LA21_0==32) ) {
                alt21=1;
            }
            switch (alt21) {
                case 1 :
                    // InternalBTree.g:1134:4: otherlv_7= 'comment' ( (lv_comment_8_0= RULE_STRING ) )
                    {
                    otherlv_7=(Token)match(input,32,FOLLOW_26); 

                    				newLeafNode(otherlv_7, grammarAccess.getBBNodeAccess().getCommentKeyword_7_0());
                    			
                    // InternalBTree.g:1138:4: ( (lv_comment_8_0= RULE_STRING ) )
                    // InternalBTree.g:1139:5: (lv_comment_8_0= RULE_STRING )
                    {
                    // InternalBTree.g:1139:5: (lv_comment_8_0= RULE_STRING )
                    // InternalBTree.g:1140:6: lv_comment_8_0= RULE_STRING
                    {
                    lv_comment_8_0=(Token)match(input,RULE_STRING,FOLLOW_34); 

                    						newLeafNode(lv_comment_8_0, grammarAccess.getBBNodeAccess().getCommentSTRINGTerminalRuleCall_7_1_0());
                    					

                    						if (current==null) {
                    							current = createModelElement(grammarAccess.getBBNodeRule());
                    						}
                    						setWithLastConsumed(
                    							current,
                    							"comment",
                    							lv_comment_8_0,
                    							"org.eclipse.xtext.common.Terminals.STRING");
                    					

                    }


                    }


                    }
                    break;

            }

            otherlv_9=(Token)match(input,23,FOLLOW_22); 

            			newLeafNode(otherlv_9, grammarAccess.getBBNodeAccess().getEndKeyword_8());
            		
            // InternalBTree.g:1161:3: (otherlv_10= ';' )?
            int alt22=2;
            int LA22_0 = input.LA(1);

            if ( (LA22_0==13) ) {
                alt22=1;
            }
            switch (alt22) {
                case 1 :
                    // InternalBTree.g:1162:4: otherlv_10= ';'
                    {
                    otherlv_10=(Token)match(input,13,FOLLOW_2); 

                    				newLeafNode(otherlv_10, grammarAccess.getBBNodeAccess().getSemicolonKeyword_9());
                    			

                    }
                    break;

            }


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBBNode"


    // $ANTLR start "entryRuleCheckNode"
    // InternalBTree.g:1171:1: entryRuleCheckNode returns [EObject current=null] : iv_ruleCheckNode= ruleCheckNode EOF ;
    public final EObject entryRuleCheckNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleCheckNode = null;


        try {
            // InternalBTree.g:1171:50: (iv_ruleCheckNode= ruleCheckNode EOF )
            // InternalBTree.g:1172:2: iv_ruleCheckNode= ruleCheckNode EOF
            {
             newCompositeNode(grammarAccess.getCheckNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleCheckNode=ruleCheckNode();

            state._fsp--;

             current =iv_ruleCheckNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleCheckNode"


    // $ANTLR start "ruleCheckNode"
    // InternalBTree.g:1178:1: ruleCheckNode returns [EObject current=null] : ( () otherlv_1= 'check' ( (lv_name_2_0= RULE_ID ) ) ( (otherlv_3= RULE_ID ) ) otherlv_4= '==' ( (lv_default_5_0= ruleBASETYPE ) ) otherlv_6= ';' ) ;
    public final EObject ruleCheckNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_1=null;
        Token lv_name_2_0=null;
        Token otherlv_3=null;
        Token otherlv_4=null;
        Token otherlv_6=null;
        AntlrDatatypeRuleToken lv_default_5_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1184:2: ( ( () otherlv_1= 'check' ( (lv_name_2_0= RULE_ID ) ) ( (otherlv_3= RULE_ID ) ) otherlv_4= '==' ( (lv_default_5_0= ruleBASETYPE ) ) otherlv_6= ';' ) )
            // InternalBTree.g:1185:2: ( () otherlv_1= 'check' ( (lv_name_2_0= RULE_ID ) ) ( (otherlv_3= RULE_ID ) ) otherlv_4= '==' ( (lv_default_5_0= ruleBASETYPE ) ) otherlv_6= ';' )
            {
            // InternalBTree.g:1185:2: ( () otherlv_1= 'check' ( (lv_name_2_0= RULE_ID ) ) ( (otherlv_3= RULE_ID ) ) otherlv_4= '==' ( (lv_default_5_0= ruleBASETYPE ) ) otherlv_6= ';' )
            // InternalBTree.g:1186:3: () otherlv_1= 'check' ( (lv_name_2_0= RULE_ID ) ) ( (otherlv_3= RULE_ID ) ) otherlv_4= '==' ( (lv_default_5_0= ruleBASETYPE ) ) otherlv_6= ';'
            {
            // InternalBTree.g:1186:3: ()
            // InternalBTree.g:1187:4: 
            {

            				current = forceCreateModelElement(
            					grammarAccess.getCheckNodeAccess().getBBVarAction_0(),
            					current);
            			

            }

            otherlv_1=(Token)match(input,33,FOLLOW_3); 

            			newLeafNode(otherlv_1, grammarAccess.getCheckNodeAccess().getCheckKeyword_1());
            		
            // InternalBTree.g:1197:3: ( (lv_name_2_0= RULE_ID ) )
            // InternalBTree.g:1198:4: (lv_name_2_0= RULE_ID )
            {
            // InternalBTree.g:1198:4: (lv_name_2_0= RULE_ID )
            // InternalBTree.g:1199:5: lv_name_2_0= RULE_ID
            {
            lv_name_2_0=(Token)match(input,RULE_ID,FOLLOW_3); 

            					newLeafNode(lv_name_2_0, grammarAccess.getCheckNodeAccess().getNameIDTerminalRuleCall_2_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getCheckNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_2_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:1215:3: ( (otherlv_3= RULE_ID ) )
            // InternalBTree.g:1216:4: (otherlv_3= RULE_ID )
            {
            // InternalBTree.g:1216:4: (otherlv_3= RULE_ID )
            // InternalBTree.g:1217:5: otherlv_3= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getCheckNodeRule());
            					}
            				
            otherlv_3=(Token)match(input,RULE_ID,FOLLOW_35); 

            					newLeafNode(otherlv_3, grammarAccess.getCheckNodeAccess().getBbvarBBVarCrossReference_3_0());
            				

            }


            }

            otherlv_4=(Token)match(input,34,FOLLOW_28); 

            			newLeafNode(otherlv_4, grammarAccess.getCheckNodeAccess().getEqualsSignEqualsSignKeyword_4());
            		
            // InternalBTree.g:1232:3: ( (lv_default_5_0= ruleBASETYPE ) )
            // InternalBTree.g:1233:4: (lv_default_5_0= ruleBASETYPE )
            {
            // InternalBTree.g:1233:4: (lv_default_5_0= ruleBASETYPE )
            // InternalBTree.g:1234:5: lv_default_5_0= ruleBASETYPE
            {

            					newCompositeNode(grammarAccess.getCheckNodeAccess().getDefaultBASETYPEParserRuleCall_5_0());
            				
            pushFollow(FOLLOW_4);
            lv_default_5_0=ruleBASETYPE();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getCheckNodeRule());
            					}
            					set(
            						current,
            						"default",
            						lv_default_5_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.BASETYPE");
            					afterParserOrEnumRuleCall();
            				

            }


            }

            otherlv_6=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_6, grammarAccess.getCheckNodeAccess().getSemicolonKeyword_6());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleCheckNode"


    // $ANTLR start "entryRuleBehaviorNode"
    // InternalBTree.g:1259:1: entryRuleBehaviorNode returns [EObject current=null] : iv_ruleBehaviorNode= ruleBehaviorNode EOF ;
    public final EObject entryRuleBehaviorNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBehaviorNode = null;


        try {
            // InternalBTree.g:1259:53: (iv_ruleBehaviorNode= ruleBehaviorNode EOF )
            // InternalBTree.g:1260:2: iv_ruleBehaviorNode= ruleBehaviorNode EOF
            {
             newCompositeNode(grammarAccess.getBehaviorNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBehaviorNode=ruleBehaviorNode();

            state._fsp--;

             current =iv_ruleBehaviorNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBehaviorNode"


    // $ANTLR start "ruleBehaviorNode"
    // InternalBTree.g:1266:1: ruleBehaviorNode returns [EObject current=null] : (this_StdBehaviorNode_0= ruleStdBehaviorNode | this_TaskNode_1= ruleTaskNode ) ;
    public final EObject ruleBehaviorNode() throws RecognitionException {
        EObject current = null;

        EObject this_StdBehaviorNode_0 = null;

        EObject this_TaskNode_1 = null;



        	enterRule();

        try {
            // InternalBTree.g:1272:2: ( (this_StdBehaviorNode_0= ruleStdBehaviorNode | this_TaskNode_1= ruleTaskNode ) )
            // InternalBTree.g:1273:2: (this_StdBehaviorNode_0= ruleStdBehaviorNode | this_TaskNode_1= ruleTaskNode )
            {
            // InternalBTree.g:1273:2: (this_StdBehaviorNode_0= ruleStdBehaviorNode | this_TaskNode_1= ruleTaskNode )
            int alt23=2;
            int LA23_0 = input.LA(1);

            if ( ((LA23_0>=35 && LA23_0<=37)) ) {
                alt23=1;
            }
            else if ( (LA23_0==38) ) {
                alt23=2;
            }
            else {
                NoViableAltException nvae =
                    new NoViableAltException("", 23, 0, input);

                throw nvae;
            }
            switch (alt23) {
                case 1 :
                    // InternalBTree.g:1274:3: this_StdBehaviorNode_0= ruleStdBehaviorNode
                    {

                    			newCompositeNode(grammarAccess.getBehaviorNodeAccess().getStdBehaviorNodeParserRuleCall_0());
                    		
                    pushFollow(FOLLOW_2);
                    this_StdBehaviorNode_0=ruleStdBehaviorNode();

                    state._fsp--;


                    			current = this_StdBehaviorNode_0;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 2 :
                    // InternalBTree.g:1283:3: this_TaskNode_1= ruleTaskNode
                    {

                    			newCompositeNode(grammarAccess.getBehaviorNodeAccess().getTaskNodeParserRuleCall_1());
                    		
                    pushFollow(FOLLOW_2);
                    this_TaskNode_1=ruleTaskNode();

                    state._fsp--;


                    			current = this_TaskNode_1;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBehaviorNode"


    // $ANTLR start "entryRuleStdBehaviorNode"
    // InternalBTree.g:1295:1: entryRuleStdBehaviorNode returns [EObject current=null] : iv_ruleStdBehaviorNode= ruleStdBehaviorNode EOF ;
    public final EObject entryRuleStdBehaviorNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleStdBehaviorNode = null;


        try {
            // InternalBTree.g:1295:56: (iv_ruleStdBehaviorNode= ruleStdBehaviorNode EOF )
            // InternalBTree.g:1296:2: iv_ruleStdBehaviorNode= ruleStdBehaviorNode EOF
            {
             newCompositeNode(grammarAccess.getStdBehaviorNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleStdBehaviorNode=ruleStdBehaviorNode();

            state._fsp--;

             current =iv_ruleStdBehaviorNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleStdBehaviorNode"


    // $ANTLR start "ruleStdBehaviorNode"
    // InternalBTree.g:1302:1: ruleStdBehaviorNode returns [EObject current=null] : ( ( (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE ) ) ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ) ;
    public final EObject ruleStdBehaviorNode() throws RecognitionException {
        EObject current = null;

        Token lv_name_1_0=null;
        Token otherlv_2=null;
        AntlrDatatypeRuleToken lv_type_0_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1308:2: ( ( ( (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE ) ) ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' ) )
            // InternalBTree.g:1309:2: ( ( (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE ) ) ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' )
            {
            // InternalBTree.g:1309:2: ( ( (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE ) ) ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';' )
            // InternalBTree.g:1310:3: ( (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE ) ) ( (lv_name_1_0= RULE_ID ) ) otherlv_2= ';'
            {
            // InternalBTree.g:1310:3: ( (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE ) )
            // InternalBTree.g:1311:4: (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE )
            {
            // InternalBTree.g:1311:4: (lv_type_0_0= ruleSTD_BEHAVIOR_TYPE )
            // InternalBTree.g:1312:5: lv_type_0_0= ruleSTD_BEHAVIOR_TYPE
            {

            					newCompositeNode(grammarAccess.getStdBehaviorNodeAccess().getTypeSTD_BEHAVIOR_TYPEParserRuleCall_0_0());
            				
            pushFollow(FOLLOW_3);
            lv_type_0_0=ruleSTD_BEHAVIOR_TYPE();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getStdBehaviorNodeRule());
            					}
            					set(
            						current,
            						"type",
            						lv_type_0_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.STD_BEHAVIOR_TYPE");
            					afterParserOrEnumRuleCall();
            				

            }


            }

            // InternalBTree.g:1329:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:1330:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:1330:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:1331:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_4); 

            					newLeafNode(lv_name_1_0, grammarAccess.getStdBehaviorNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getStdBehaviorNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            otherlv_2=(Token)match(input,13,FOLLOW_2); 

            			newLeafNode(otherlv_2, grammarAccess.getStdBehaviorNodeAccess().getSemicolonKeyword_2());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleStdBehaviorNode"


    // $ANTLR start "entryRuleSTD_BEHAVIOR_TYPE"
    // InternalBTree.g:1355:1: entryRuleSTD_BEHAVIOR_TYPE returns [String current=null] : iv_ruleSTD_BEHAVIOR_TYPE= ruleSTD_BEHAVIOR_TYPE EOF ;
    public final String entryRuleSTD_BEHAVIOR_TYPE() throws RecognitionException {
        String current = null;

        AntlrDatatypeRuleToken iv_ruleSTD_BEHAVIOR_TYPE = null;


        try {
            // InternalBTree.g:1355:57: (iv_ruleSTD_BEHAVIOR_TYPE= ruleSTD_BEHAVIOR_TYPE EOF )
            // InternalBTree.g:1356:2: iv_ruleSTD_BEHAVIOR_TYPE= ruleSTD_BEHAVIOR_TYPE EOF
            {
             newCompositeNode(grammarAccess.getSTD_BEHAVIOR_TYPERule()); 
            pushFollow(FOLLOW_1);
            iv_ruleSTD_BEHAVIOR_TYPE=ruleSTD_BEHAVIOR_TYPE();

            state._fsp--;

             current =iv_ruleSTD_BEHAVIOR_TYPE.getText(); 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleSTD_BEHAVIOR_TYPE"


    // $ANTLR start "ruleSTD_BEHAVIOR_TYPE"
    // InternalBTree.g:1362:1: ruleSTD_BEHAVIOR_TYPE returns [AntlrDatatypeRuleToken current=new AntlrDatatypeRuleToken()] : (kw= 'success' | kw= 'failure' | kw= 'running' ) ;
    public final AntlrDatatypeRuleToken ruleSTD_BEHAVIOR_TYPE() throws RecognitionException {
        AntlrDatatypeRuleToken current = new AntlrDatatypeRuleToken();

        Token kw=null;


        	enterRule();

        try {
            // InternalBTree.g:1368:2: ( (kw= 'success' | kw= 'failure' | kw= 'running' ) )
            // InternalBTree.g:1369:2: (kw= 'success' | kw= 'failure' | kw= 'running' )
            {
            // InternalBTree.g:1369:2: (kw= 'success' | kw= 'failure' | kw= 'running' )
            int alt24=3;
            switch ( input.LA(1) ) {
            case 35:
                {
                alt24=1;
                }
                break;
            case 36:
                {
                alt24=2;
                }
                break;
            case 37:
                {
                alt24=3;
                }
                break;
            default:
                NoViableAltException nvae =
                    new NoViableAltException("", 24, 0, input);

                throw nvae;
            }

            switch (alt24) {
                case 1 :
                    // InternalBTree.g:1370:3: kw= 'success'
                    {
                    kw=(Token)match(input,35,FOLLOW_2); 

                    			current.merge(kw);
                    			newLeafNode(kw, grammarAccess.getSTD_BEHAVIOR_TYPEAccess().getSuccessKeyword_0());
                    		

                    }
                    break;
                case 2 :
                    // InternalBTree.g:1376:3: kw= 'failure'
                    {
                    kw=(Token)match(input,36,FOLLOW_2); 

                    			current.merge(kw);
                    			newLeafNode(kw, grammarAccess.getSTD_BEHAVIOR_TYPEAccess().getFailureKeyword_1());
                    		

                    }
                    break;
                case 3 :
                    // InternalBTree.g:1382:3: kw= 'running'
                    {
                    kw=(Token)match(input,37,FOLLOW_2); 

                    			current.merge(kw);
                    			newLeafNode(kw, grammarAccess.getSTD_BEHAVIOR_TYPEAccess().getRunningKeyword_2());
                    		

                    }
                    break;

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleSTD_BEHAVIOR_TYPE"


    // $ANTLR start "entryRuleTaskNode"
    // InternalBTree.g:1391:1: entryRuleTaskNode returns [EObject current=null] : iv_ruleTaskNode= ruleTaskNode EOF ;
    public final EObject entryRuleTaskNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleTaskNode = null;


        try {
            // InternalBTree.g:1391:49: (iv_ruleTaskNode= ruleTaskNode EOF )
            // InternalBTree.g:1392:2: iv_ruleTaskNode= ruleTaskNode EOF
            {
             newCompositeNode(grammarAccess.getTaskNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleTaskNode=ruleTaskNode();

            state._fsp--;

             current =iv_ruleTaskNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleTaskNode"


    // $ANTLR start "ruleTaskNode"
    // InternalBTree.g:1398:1: ruleTaskNode returns [EObject current=null] : (otherlv_0= 'task' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= 'in' ( (lv_input_topics_3_0= ruleTopicArg ) ) (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )* otherlv_6= ';' )? (otherlv_7= 'out' ( (lv_output_topics_8_0= ruleTopicArg ) ) (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )* otherlv_11= ';' )? ( (lv_bb_vars_12_0= ruleBBVar ) )* ( (lv_args_13_0= ruleArg ) )* (otherlv_14= 'comment' ( (lv_comment_15_0= RULE_STRING ) ) (otherlv_16= ';' )? )? otherlv_17= 'end' (otherlv_18= ';' )? ) ;
    public final EObject ruleTaskNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_4=null;
        Token otherlv_6=null;
        Token otherlv_7=null;
        Token otherlv_9=null;
        Token otherlv_11=null;
        Token otherlv_14=null;
        Token lv_comment_15_0=null;
        Token otherlv_16=null;
        Token otherlv_17=null;
        Token otherlv_18=null;
        EObject lv_input_topics_3_0 = null;

        EObject lv_input_topics_5_0 = null;

        EObject lv_output_topics_8_0 = null;

        EObject lv_output_topics_10_0 = null;

        EObject lv_bb_vars_12_0 = null;

        EObject lv_args_13_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1404:2: ( (otherlv_0= 'task' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= 'in' ( (lv_input_topics_3_0= ruleTopicArg ) ) (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )* otherlv_6= ';' )? (otherlv_7= 'out' ( (lv_output_topics_8_0= ruleTopicArg ) ) (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )* otherlv_11= ';' )? ( (lv_bb_vars_12_0= ruleBBVar ) )* ( (lv_args_13_0= ruleArg ) )* (otherlv_14= 'comment' ( (lv_comment_15_0= RULE_STRING ) ) (otherlv_16= ';' )? )? otherlv_17= 'end' (otherlv_18= ';' )? ) )
            // InternalBTree.g:1405:2: (otherlv_0= 'task' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= 'in' ( (lv_input_topics_3_0= ruleTopicArg ) ) (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )* otherlv_6= ';' )? (otherlv_7= 'out' ( (lv_output_topics_8_0= ruleTopicArg ) ) (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )* otherlv_11= ';' )? ( (lv_bb_vars_12_0= ruleBBVar ) )* ( (lv_args_13_0= ruleArg ) )* (otherlv_14= 'comment' ( (lv_comment_15_0= RULE_STRING ) ) (otherlv_16= ';' )? )? otherlv_17= 'end' (otherlv_18= ';' )? )
            {
            // InternalBTree.g:1405:2: (otherlv_0= 'task' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= 'in' ( (lv_input_topics_3_0= ruleTopicArg ) ) (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )* otherlv_6= ';' )? (otherlv_7= 'out' ( (lv_output_topics_8_0= ruleTopicArg ) ) (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )* otherlv_11= ';' )? ( (lv_bb_vars_12_0= ruleBBVar ) )* ( (lv_args_13_0= ruleArg ) )* (otherlv_14= 'comment' ( (lv_comment_15_0= RULE_STRING ) ) (otherlv_16= ';' )? )? otherlv_17= 'end' (otherlv_18= ';' )? )
            // InternalBTree.g:1406:3: otherlv_0= 'task' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= 'in' ( (lv_input_topics_3_0= ruleTopicArg ) ) (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )* otherlv_6= ';' )? (otherlv_7= 'out' ( (lv_output_topics_8_0= ruleTopicArg ) ) (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )* otherlv_11= ';' )? ( (lv_bb_vars_12_0= ruleBBVar ) )* ( (lv_args_13_0= ruleArg ) )* (otherlv_14= 'comment' ( (lv_comment_15_0= RULE_STRING ) ) (otherlv_16= ';' )? )? otherlv_17= 'end' (otherlv_18= ';' )?
            {
            otherlv_0=(Token)match(input,38,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getTaskNodeAccess().getTaskKeyword_0());
            		
            // InternalBTree.g:1410:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:1411:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:1411:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:1412:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_36); 

            					newLeafNode(lv_name_1_0, grammarAccess.getTaskNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTaskNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:1428:3: (otherlv_2= 'in' ( (lv_input_topics_3_0= ruleTopicArg ) ) (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )* otherlv_6= ';' )?
            int alt26=2;
            int LA26_0 = input.LA(1);

            if ( (LA26_0==39) ) {
                alt26=1;
            }
            switch (alt26) {
                case 1 :
                    // InternalBTree.g:1429:4: otherlv_2= 'in' ( (lv_input_topics_3_0= ruleTopicArg ) ) (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )* otherlv_6= ';'
                    {
                    otherlv_2=(Token)match(input,39,FOLLOW_3); 

                    				newLeafNode(otherlv_2, grammarAccess.getTaskNodeAccess().getInKeyword_2_0());
                    			
                    // InternalBTree.g:1433:4: ( (lv_input_topics_3_0= ruleTopicArg ) )
                    // InternalBTree.g:1434:5: (lv_input_topics_3_0= ruleTopicArg )
                    {
                    // InternalBTree.g:1434:5: (lv_input_topics_3_0= ruleTopicArg )
                    // InternalBTree.g:1435:6: lv_input_topics_3_0= ruleTopicArg
                    {

                    						newCompositeNode(grammarAccess.getTaskNodeAccess().getInput_topicsTopicArgParserRuleCall_2_1_0());
                    					
                    pushFollow(FOLLOW_37);
                    lv_input_topics_3_0=ruleTopicArg();

                    state._fsp--;


                    						if (current==null) {
                    							current = createModelElementForParent(grammarAccess.getTaskNodeRule());
                    						}
                    						add(
                    							current,
                    							"input_topics",
                    							lv_input_topics_3_0,
                    							"edu.vanderbilt.isis.alc.btree.BTree.TopicArg");
                    						afterParserOrEnumRuleCall();
                    					

                    }


                    }

                    // InternalBTree.g:1452:4: (otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) ) )*
                    loop25:
                    do {
                        int alt25=2;
                        int LA25_0 = input.LA(1);

                        if ( (LA25_0==18) ) {
                            alt25=1;
                        }


                        switch (alt25) {
                    	case 1 :
                    	    // InternalBTree.g:1453:5: otherlv_4= ',' ( (lv_input_topics_5_0= ruleTopicArg ) )
                    	    {
                    	    otherlv_4=(Token)match(input,18,FOLLOW_3); 

                    	    					newLeafNode(otherlv_4, grammarAccess.getTaskNodeAccess().getCommaKeyword_2_2_0());
                    	    				
                    	    // InternalBTree.g:1457:5: ( (lv_input_topics_5_0= ruleTopicArg ) )
                    	    // InternalBTree.g:1458:6: (lv_input_topics_5_0= ruleTopicArg )
                    	    {
                    	    // InternalBTree.g:1458:6: (lv_input_topics_5_0= ruleTopicArg )
                    	    // InternalBTree.g:1459:7: lv_input_topics_5_0= ruleTopicArg
                    	    {

                    	    							newCompositeNode(grammarAccess.getTaskNodeAccess().getInput_topicsTopicArgParserRuleCall_2_2_1_0());
                    	    						
                    	    pushFollow(FOLLOW_37);
                    	    lv_input_topics_5_0=ruleTopicArg();

                    	    state._fsp--;


                    	    							if (current==null) {
                    	    								current = createModelElementForParent(grammarAccess.getTaskNodeRule());
                    	    							}
                    	    							add(
                    	    								current,
                    	    								"input_topics",
                    	    								lv_input_topics_5_0,
                    	    								"edu.vanderbilt.isis.alc.btree.BTree.TopicArg");
                    	    							afterParserOrEnumRuleCall();
                    	    						

                    	    }


                    	    }


                    	    }
                    	    break;

                    	default :
                    	    break loop25;
                        }
                    } while (true);

                    otherlv_6=(Token)match(input,13,FOLLOW_38); 

                    				newLeafNode(otherlv_6, grammarAccess.getTaskNodeAccess().getSemicolonKeyword_2_3());
                    			

                    }
                    break;

            }

            // InternalBTree.g:1482:3: (otherlv_7= 'out' ( (lv_output_topics_8_0= ruleTopicArg ) ) (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )* otherlv_11= ';' )?
            int alt28=2;
            int LA28_0 = input.LA(1);

            if ( (LA28_0==40) ) {
                alt28=1;
            }
            switch (alt28) {
                case 1 :
                    // InternalBTree.g:1483:4: otherlv_7= 'out' ( (lv_output_topics_8_0= ruleTopicArg ) ) (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )* otherlv_11= ';'
                    {
                    otherlv_7=(Token)match(input,40,FOLLOW_3); 

                    				newLeafNode(otherlv_7, grammarAccess.getTaskNodeAccess().getOutKeyword_3_0());
                    			
                    // InternalBTree.g:1487:4: ( (lv_output_topics_8_0= ruleTopicArg ) )
                    // InternalBTree.g:1488:5: (lv_output_topics_8_0= ruleTopicArg )
                    {
                    // InternalBTree.g:1488:5: (lv_output_topics_8_0= ruleTopicArg )
                    // InternalBTree.g:1489:6: lv_output_topics_8_0= ruleTopicArg
                    {

                    						newCompositeNode(grammarAccess.getTaskNodeAccess().getOutput_topicsTopicArgParserRuleCall_3_1_0());
                    					
                    pushFollow(FOLLOW_37);
                    lv_output_topics_8_0=ruleTopicArg();

                    state._fsp--;


                    						if (current==null) {
                    							current = createModelElementForParent(grammarAccess.getTaskNodeRule());
                    						}
                    						add(
                    							current,
                    							"output_topics",
                    							lv_output_topics_8_0,
                    							"edu.vanderbilt.isis.alc.btree.BTree.TopicArg");
                    						afterParserOrEnumRuleCall();
                    					

                    }


                    }

                    // InternalBTree.g:1506:4: (otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) ) )*
                    loop27:
                    do {
                        int alt27=2;
                        int LA27_0 = input.LA(1);

                        if ( (LA27_0==18) ) {
                            alt27=1;
                        }


                        switch (alt27) {
                    	case 1 :
                    	    // InternalBTree.g:1507:5: otherlv_9= ',' ( (lv_output_topics_10_0= ruleTopicArg ) )
                    	    {
                    	    otherlv_9=(Token)match(input,18,FOLLOW_3); 

                    	    					newLeafNode(otherlv_9, grammarAccess.getTaskNodeAccess().getCommaKeyword_3_2_0());
                    	    				
                    	    // InternalBTree.g:1511:5: ( (lv_output_topics_10_0= ruleTopicArg ) )
                    	    // InternalBTree.g:1512:6: (lv_output_topics_10_0= ruleTopicArg )
                    	    {
                    	    // InternalBTree.g:1512:6: (lv_output_topics_10_0= ruleTopicArg )
                    	    // InternalBTree.g:1513:7: lv_output_topics_10_0= ruleTopicArg
                    	    {

                    	    							newCompositeNode(grammarAccess.getTaskNodeAccess().getOutput_topicsTopicArgParserRuleCall_3_2_1_0());
                    	    						
                    	    pushFollow(FOLLOW_37);
                    	    lv_output_topics_10_0=ruleTopicArg();

                    	    state._fsp--;


                    	    							if (current==null) {
                    	    								current = createModelElementForParent(grammarAccess.getTaskNodeRule());
                    	    							}
                    	    							add(
                    	    								current,
                    	    								"output_topics",
                    	    								lv_output_topics_10_0,
                    	    								"edu.vanderbilt.isis.alc.btree.BTree.TopicArg");
                    	    							afterParserOrEnumRuleCall();
                    	    						

                    	    }


                    	    }


                    	    }
                    	    break;

                    	default :
                    	    break loop27;
                        }
                    } while (true);

                    otherlv_11=(Token)match(input,13,FOLLOW_32); 

                    				newLeafNode(otherlv_11, grammarAccess.getTaskNodeAccess().getSemicolonKeyword_3_3());
                    			

                    }
                    break;

            }

            // InternalBTree.g:1536:3: ( (lv_bb_vars_12_0= ruleBBVar ) )*
            loop29:
            do {
                int alt29=2;
                int LA29_0 = input.LA(1);

                if ( (LA29_0==27) ) {
                    alt29=1;
                }


                switch (alt29) {
            	case 1 :
            	    // InternalBTree.g:1537:4: (lv_bb_vars_12_0= ruleBBVar )
            	    {
            	    // InternalBTree.g:1537:4: (lv_bb_vars_12_0= ruleBBVar )
            	    // InternalBTree.g:1538:5: lv_bb_vars_12_0= ruleBBVar
            	    {

            	    					newCompositeNode(grammarAccess.getTaskNodeAccess().getBb_varsBBVarParserRuleCall_4_0());
            	    				
            	    pushFollow(FOLLOW_32);
            	    lv_bb_vars_12_0=ruleBBVar();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getTaskNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"bb_vars",
            	    						lv_bb_vars_12_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.BBVar");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop29;
                }
            } while (true);

            // InternalBTree.g:1555:3: ( (lv_args_13_0= ruleArg ) )*
            loop30:
            do {
                int alt30=2;
                int LA30_0 = input.LA(1);

                if ( (LA30_0==29) ) {
                    alt30=1;
                }


                switch (alt30) {
            	case 1 :
            	    // InternalBTree.g:1556:4: (lv_args_13_0= ruleArg )
            	    {
            	    // InternalBTree.g:1556:4: (lv_args_13_0= ruleArg )
            	    // InternalBTree.g:1557:5: lv_args_13_0= ruleArg
            	    {

            	    					newCompositeNode(grammarAccess.getTaskNodeAccess().getArgsArgParserRuleCall_5_0());
            	    				
            	    pushFollow(FOLLOW_33);
            	    lv_args_13_0=ruleArg();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getTaskNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"args",
            	    						lv_args_13_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.Arg");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop30;
                }
            } while (true);

            // InternalBTree.g:1574:3: (otherlv_14= 'comment' ( (lv_comment_15_0= RULE_STRING ) ) (otherlv_16= ';' )? )?
            int alt32=2;
            int LA32_0 = input.LA(1);

            if ( (LA32_0==32) ) {
                alt32=1;
            }
            switch (alt32) {
                case 1 :
                    // InternalBTree.g:1575:4: otherlv_14= 'comment' ( (lv_comment_15_0= RULE_STRING ) ) (otherlv_16= ';' )?
                    {
                    otherlv_14=(Token)match(input,32,FOLLOW_26); 

                    				newLeafNode(otherlv_14, grammarAccess.getTaskNodeAccess().getCommentKeyword_6_0());
                    			
                    // InternalBTree.g:1579:4: ( (lv_comment_15_0= RULE_STRING ) )
                    // InternalBTree.g:1580:5: (lv_comment_15_0= RULE_STRING )
                    {
                    // InternalBTree.g:1580:5: (lv_comment_15_0= RULE_STRING )
                    // InternalBTree.g:1581:6: lv_comment_15_0= RULE_STRING
                    {
                    lv_comment_15_0=(Token)match(input,RULE_STRING,FOLLOW_39); 

                    						newLeafNode(lv_comment_15_0, grammarAccess.getTaskNodeAccess().getCommentSTRINGTerminalRuleCall_6_1_0());
                    					

                    						if (current==null) {
                    							current = createModelElement(grammarAccess.getTaskNodeRule());
                    						}
                    						setWithLastConsumed(
                    							current,
                    							"comment",
                    							lv_comment_15_0,
                    							"org.eclipse.xtext.common.Terminals.STRING");
                    					

                    }


                    }

                    // InternalBTree.g:1597:4: (otherlv_16= ';' )?
                    int alt31=2;
                    int LA31_0 = input.LA(1);

                    if ( (LA31_0==13) ) {
                        alt31=1;
                    }
                    switch (alt31) {
                        case 1 :
                            // InternalBTree.g:1598:5: otherlv_16= ';'
                            {
                            otherlv_16=(Token)match(input,13,FOLLOW_34); 

                            					newLeafNode(otherlv_16, grammarAccess.getTaskNodeAccess().getSemicolonKeyword_6_2());
                            				

                            }
                            break;

                    }


                    }
                    break;

            }

            otherlv_17=(Token)match(input,23,FOLLOW_22); 

            			newLeafNode(otherlv_17, grammarAccess.getTaskNodeAccess().getEndKeyword_7());
            		
            // InternalBTree.g:1608:3: (otherlv_18= ';' )?
            int alt33=2;
            int LA33_0 = input.LA(1);

            if ( (LA33_0==13) ) {
                alt33=1;
            }
            switch (alt33) {
                case 1 :
                    // InternalBTree.g:1609:4: otherlv_18= ';'
                    {
                    otherlv_18=(Token)match(input,13,FOLLOW_2); 

                    				newLeafNode(otherlv_18, grammarAccess.getTaskNodeAccess().getSemicolonKeyword_8());
                    			

                    }
                    break;

            }


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleTaskNode"


    // $ANTLR start "entryRuleTopicArg"
    // InternalBTree.g:1618:1: entryRuleTopicArg returns [EObject current=null] : iv_ruleTopicArg= ruleTopicArg EOF ;
    public final EObject entryRuleTopicArg() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleTopicArg = null;


        try {
            // InternalBTree.g:1618:49: (iv_ruleTopicArg= ruleTopicArg EOF )
            // InternalBTree.g:1619:2: iv_ruleTopicArg= ruleTopicArg EOF
            {
             newCompositeNode(grammarAccess.getTopicArgRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleTopicArg=ruleTopicArg();

            state._fsp--;

             current =iv_ruleTopicArg; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleTopicArg"


    // $ANTLR start "ruleTopicArg"
    // InternalBTree.g:1625:1: ruleTopicArg returns [EObject current=null] : ( ( (otherlv_0= RULE_ID ) ) ( (lv_name_1_0= RULE_ID ) ) ) ;
    public final EObject ruleTopicArg() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;


        	enterRule();

        try {
            // InternalBTree.g:1631:2: ( ( ( (otherlv_0= RULE_ID ) ) ( (lv_name_1_0= RULE_ID ) ) ) )
            // InternalBTree.g:1632:2: ( ( (otherlv_0= RULE_ID ) ) ( (lv_name_1_0= RULE_ID ) ) )
            {
            // InternalBTree.g:1632:2: ( ( (otherlv_0= RULE_ID ) ) ( (lv_name_1_0= RULE_ID ) ) )
            // InternalBTree.g:1633:3: ( (otherlv_0= RULE_ID ) ) ( (lv_name_1_0= RULE_ID ) )
            {
            // InternalBTree.g:1633:3: ( (otherlv_0= RULE_ID ) )
            // InternalBTree.g:1634:4: (otherlv_0= RULE_ID )
            {
            // InternalBTree.g:1634:4: (otherlv_0= RULE_ID )
            // InternalBTree.g:1635:5: otherlv_0= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTopicArgRule());
            					}
            				
            otherlv_0=(Token)match(input,RULE_ID,FOLLOW_3); 

            					newLeafNode(otherlv_0, grammarAccess.getTopicArgAccess().getTypeTopicCrossReference_0_0());
            				

            }


            }

            // InternalBTree.g:1646:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:1647:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:1647:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:1648:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_2); 

            					newLeafNode(lv_name_1_0, grammarAccess.getTopicArgAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTopicArgRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleTopicArg"


    // $ANTLR start "entryRuleBTree"
    // InternalBTree.g:1668:1: entryRuleBTree returns [EObject current=null] : iv_ruleBTree= ruleBTree EOF ;
    public final EObject entryRuleBTree() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBTree = null;


        try {
            // InternalBTree.g:1668:46: (iv_ruleBTree= ruleBTree EOF )
            // InternalBTree.g:1669:2: iv_ruleBTree= ruleBTree EOF
            {
             newCompositeNode(grammarAccess.getBTreeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBTree=ruleBTree();

            state._fsp--;

             current =iv_ruleBTree; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBTree"


    // $ANTLR start "ruleBTree"
    // InternalBTree.g:1675:1: ruleBTree returns [EObject current=null] : ( (lv_btree_0_0= ruleBTreeNode ) ) ;
    public final EObject ruleBTree() throws RecognitionException {
        EObject current = null;

        EObject lv_btree_0_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1681:2: ( ( (lv_btree_0_0= ruleBTreeNode ) ) )
            // InternalBTree.g:1682:2: ( (lv_btree_0_0= ruleBTreeNode ) )
            {
            // InternalBTree.g:1682:2: ( (lv_btree_0_0= ruleBTreeNode ) )
            // InternalBTree.g:1683:3: (lv_btree_0_0= ruleBTreeNode )
            {
            // InternalBTree.g:1683:3: (lv_btree_0_0= ruleBTreeNode )
            // InternalBTree.g:1684:4: lv_btree_0_0= ruleBTreeNode
            {

            				newCompositeNode(grammarAccess.getBTreeAccess().getBtreeBTreeNodeParserRuleCall_0());
            			
            pushFollow(FOLLOW_2);
            lv_btree_0_0=ruleBTreeNode();

            state._fsp--;


            				if (current==null) {
            					current = createModelElementForParent(grammarAccess.getBTreeRule());
            				}
            				set(
            					current,
            					"btree",
            					lv_btree_0_0,
            					"edu.vanderbilt.isis.alc.btree.BTree.BTreeNode");
            				afterParserOrEnumRuleCall();
            			

            }


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBTree"


    // $ANTLR start "entryRuleBTreeNode"
    // InternalBTree.g:1704:1: entryRuleBTreeNode returns [EObject current=null] : iv_ruleBTreeNode= ruleBTreeNode EOF ;
    public final EObject entryRuleBTreeNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleBTreeNode = null;


        try {
            // InternalBTree.g:1704:50: (iv_ruleBTreeNode= ruleBTreeNode EOF )
            // InternalBTree.g:1705:2: iv_ruleBTreeNode= ruleBTreeNode EOF
            {
             newCompositeNode(grammarAccess.getBTreeNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBTreeNode=ruleBTreeNode();

            state._fsp--;

             current =iv_ruleBTreeNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBTreeNode"


    // $ANTLR start "ruleBTreeNode"
    // InternalBTree.g:1711:1: ruleBTreeNode returns [EObject current=null] : (this_ParBTNode_0= ruleParBTNode | this_SeqBTNode_1= ruleSeqBTNode | this_SelBTNode_2= ruleSelBTNode | this_SIFBTNode_3= ruleSIFBTNode | this_MonBTNode_4= ruleMonBTNode | this_TaskBTNode_5= ruleTaskBTNode | this_TimerBTNode_6= ruleTimerBTNode | this_CheckBTNode_7= ruleCheckBTNode ) ;
    public final EObject ruleBTreeNode() throws RecognitionException {
        EObject current = null;

        EObject this_ParBTNode_0 = null;

        EObject this_SeqBTNode_1 = null;

        EObject this_SelBTNode_2 = null;

        EObject this_SIFBTNode_3 = null;

        EObject this_MonBTNode_4 = null;

        EObject this_TaskBTNode_5 = null;

        EObject this_TimerBTNode_6 = null;

        EObject this_CheckBTNode_7 = null;



        	enterRule();

        try {
            // InternalBTree.g:1717:2: ( (this_ParBTNode_0= ruleParBTNode | this_SeqBTNode_1= ruleSeqBTNode | this_SelBTNode_2= ruleSelBTNode | this_SIFBTNode_3= ruleSIFBTNode | this_MonBTNode_4= ruleMonBTNode | this_TaskBTNode_5= ruleTaskBTNode | this_TimerBTNode_6= ruleTimerBTNode | this_CheckBTNode_7= ruleCheckBTNode ) )
            // InternalBTree.g:1718:2: (this_ParBTNode_0= ruleParBTNode | this_SeqBTNode_1= ruleSeqBTNode | this_SelBTNode_2= ruleSelBTNode | this_SIFBTNode_3= ruleSIFBTNode | this_MonBTNode_4= ruleMonBTNode | this_TaskBTNode_5= ruleTaskBTNode | this_TimerBTNode_6= ruleTimerBTNode | this_CheckBTNode_7= ruleCheckBTNode )
            {
            // InternalBTree.g:1718:2: (this_ParBTNode_0= ruleParBTNode | this_SeqBTNode_1= ruleSeqBTNode | this_SelBTNode_2= ruleSelBTNode | this_SIFBTNode_3= ruleSIFBTNode | this_MonBTNode_4= ruleMonBTNode | this_TaskBTNode_5= ruleTaskBTNode | this_TimerBTNode_6= ruleTimerBTNode | this_CheckBTNode_7= ruleCheckBTNode )
            int alt34=8;
            switch ( input.LA(1) ) {
            case 41:
                {
                alt34=1;
                }
                break;
            case 44:
                {
                alt34=2;
                }
                break;
            case 45:
                {
                alt34=3;
                }
                break;
            case 46:
                {
                alt34=4;
                }
                break;
            case 49:
                {
                alt34=5;
                }
                break;
            case 50:
                {
                alt34=6;
                }
                break;
            case 51:
                {
                alt34=7;
                }
                break;
            case 52:
                {
                alt34=8;
                }
                break;
            default:
                NoViableAltException nvae =
                    new NoViableAltException("", 34, 0, input);

                throw nvae;
            }

            switch (alt34) {
                case 1 :
                    // InternalBTree.g:1719:3: this_ParBTNode_0= ruleParBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getParBTNodeParserRuleCall_0());
                    		
                    pushFollow(FOLLOW_2);
                    this_ParBTNode_0=ruleParBTNode();

                    state._fsp--;


                    			current = this_ParBTNode_0;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 2 :
                    // InternalBTree.g:1728:3: this_SeqBTNode_1= ruleSeqBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getSeqBTNodeParserRuleCall_1());
                    		
                    pushFollow(FOLLOW_2);
                    this_SeqBTNode_1=ruleSeqBTNode();

                    state._fsp--;


                    			current = this_SeqBTNode_1;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 3 :
                    // InternalBTree.g:1737:3: this_SelBTNode_2= ruleSelBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getSelBTNodeParserRuleCall_2());
                    		
                    pushFollow(FOLLOW_2);
                    this_SelBTNode_2=ruleSelBTNode();

                    state._fsp--;


                    			current = this_SelBTNode_2;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 4 :
                    // InternalBTree.g:1746:3: this_SIFBTNode_3= ruleSIFBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getSIFBTNodeParserRuleCall_3());
                    		
                    pushFollow(FOLLOW_2);
                    this_SIFBTNode_3=ruleSIFBTNode();

                    state._fsp--;


                    			current = this_SIFBTNode_3;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 5 :
                    // InternalBTree.g:1755:3: this_MonBTNode_4= ruleMonBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getMonBTNodeParserRuleCall_4());
                    		
                    pushFollow(FOLLOW_2);
                    this_MonBTNode_4=ruleMonBTNode();

                    state._fsp--;


                    			current = this_MonBTNode_4;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 6 :
                    // InternalBTree.g:1764:3: this_TaskBTNode_5= ruleTaskBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getTaskBTNodeParserRuleCall_5());
                    		
                    pushFollow(FOLLOW_2);
                    this_TaskBTNode_5=ruleTaskBTNode();

                    state._fsp--;


                    			current = this_TaskBTNode_5;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 7 :
                    // InternalBTree.g:1773:3: this_TimerBTNode_6= ruleTimerBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getTimerBTNodeParserRuleCall_6());
                    		
                    pushFollow(FOLLOW_2);
                    this_TimerBTNode_6=ruleTimerBTNode();

                    state._fsp--;


                    			current = this_TimerBTNode_6;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 8 :
                    // InternalBTree.g:1782:3: this_CheckBTNode_7= ruleCheckBTNode
                    {

                    			newCompositeNode(grammarAccess.getBTreeNodeAccess().getCheckBTNodeParserRuleCall_7());
                    		
                    pushFollow(FOLLOW_2);
                    this_CheckBTNode_7=ruleCheckBTNode();

                    state._fsp--;


                    			current = this_CheckBTNode_7;
                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBTreeNode"


    // $ANTLR start "entryRuleChildNode"
    // InternalBTree.g:1794:1: entryRuleChildNode returns [EObject current=null] : iv_ruleChildNode= ruleChildNode EOF ;
    public final EObject entryRuleChildNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleChildNode = null;


        try {
            // InternalBTree.g:1794:50: (iv_ruleChildNode= ruleChildNode EOF )
            // InternalBTree.g:1795:2: iv_ruleChildNode= ruleChildNode EOF
            {
             newCompositeNode(grammarAccess.getChildNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleChildNode=ruleChildNode();

            state._fsp--;

             current =iv_ruleChildNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleChildNode"


    // $ANTLR start "ruleChildNode"
    // InternalBTree.g:1801:1: ruleChildNode returns [EObject current=null] : this_BTreeNode_0= ruleBTreeNode ;
    public final EObject ruleChildNode() throws RecognitionException {
        EObject current = null;

        EObject this_BTreeNode_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1807:2: (this_BTreeNode_0= ruleBTreeNode )
            // InternalBTree.g:1808:2: this_BTreeNode_0= ruleBTreeNode
            {

            		newCompositeNode(grammarAccess.getChildNodeAccess().getBTreeNodeParserRuleCall());
            	
            pushFollow(FOLLOW_2);
            this_BTreeNode_0=ruleBTreeNode();

            state._fsp--;


            		current = this_BTreeNode_0;
            		afterParserOrEnumRuleCall();
            	

            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleChildNode"


    // $ANTLR start "entryRuleParBTNode"
    // InternalBTree.g:1819:1: entryRuleParBTNode returns [EObject current=null] : iv_ruleParBTNode= ruleParBTNode EOF ;
    public final EObject entryRuleParBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleParBTNode = null;


        try {
            // InternalBTree.g:1819:50: (iv_ruleParBTNode= ruleParBTNode EOF )
            // InternalBTree.g:1820:2: iv_ruleParBTNode= ruleParBTNode EOF
            {
             newCompositeNode(grammarAccess.getParBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleParBTNode=ruleParBTNode();

            state._fsp--;

             current =iv_ruleParBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleParBTNode"


    // $ANTLR start "ruleParBTNode"
    // InternalBTree.g:1826:1: ruleParBTNode returns [EObject current=null] : (otherlv_0= 'par' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' ) ;
    public final EObject ruleParBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_4=null;
        Token otherlv_5=null;
        Token otherlv_7=null;
        AntlrDatatypeRuleToken lv_cond_3_0 = null;

        EObject lv_nodes_6_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1832:2: ( (otherlv_0= 'par' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' ) )
            // InternalBTree.g:1833:2: (otherlv_0= 'par' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' )
            {
            // InternalBTree.g:1833:2: (otherlv_0= 'par' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' )
            // InternalBTree.g:1834:3: otherlv_0= 'par' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}'
            {
            otherlv_0=(Token)match(input,41,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getParBTNodeAccess().getParKeyword_0());
            		
            // InternalBTree.g:1838:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:1839:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:1839:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:1840:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_40); 

            					newLeafNode(lv_name_1_0, grammarAccess.getParBTNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getParBTNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:1856:3: (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )?
            int alt35=2;
            int LA35_0 = input.LA(1);

            if ( (LA35_0==15) ) {
                alt35=1;
            }
            switch (alt35) {
                case 1 :
                    // InternalBTree.g:1857:4: otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')'
                    {
                    otherlv_2=(Token)match(input,15,FOLLOW_41); 

                    				newLeafNode(otherlv_2, grammarAccess.getParBTNodeAccess().getLeftParenthesisKeyword_2_0());
                    			
                    // InternalBTree.g:1861:4: ( (lv_cond_3_0= ruleStatus ) )
                    // InternalBTree.g:1862:5: (lv_cond_3_0= ruleStatus )
                    {
                    // InternalBTree.g:1862:5: (lv_cond_3_0= ruleStatus )
                    // InternalBTree.g:1863:6: lv_cond_3_0= ruleStatus
                    {

                    						newCompositeNode(grammarAccess.getParBTNodeAccess().getCondStatusParserRuleCall_2_1_0());
                    					
                    pushFollow(FOLLOW_19);
                    lv_cond_3_0=ruleStatus();

                    state._fsp--;


                    						if (current==null) {
                    							current = createModelElementForParent(grammarAccess.getParBTNodeRule());
                    						}
                    						set(
                    							current,
                    							"cond",
                    							lv_cond_3_0,
                    							"edu.vanderbilt.isis.alc.btree.BTree.Status");
                    						afterParserOrEnumRuleCall();
                    					

                    }


                    }

                    otherlv_4=(Token)match(input,20,FOLLOW_42); 

                    				newLeafNode(otherlv_4, grammarAccess.getParBTNodeAccess().getRightParenthesisKeyword_2_2());
                    			

                    }
                    break;

            }

            otherlv_5=(Token)match(input,42,FOLLOW_43); 

            			newLeafNode(otherlv_5, grammarAccess.getParBTNodeAccess().getLeftCurlyBracketKeyword_3());
            		
            // InternalBTree.g:1889:3: ( (lv_nodes_6_0= ruleChildNode ) )*
            loop36:
            do {
                int alt36=2;
                int LA36_0 = input.LA(1);

                if ( (LA36_0==41||(LA36_0>=44 && LA36_0<=46)||(LA36_0>=49 && LA36_0<=52)) ) {
                    alt36=1;
                }


                switch (alt36) {
            	case 1 :
            	    // InternalBTree.g:1890:4: (lv_nodes_6_0= ruleChildNode )
            	    {
            	    // InternalBTree.g:1890:4: (lv_nodes_6_0= ruleChildNode )
            	    // InternalBTree.g:1891:5: lv_nodes_6_0= ruleChildNode
            	    {

            	    					newCompositeNode(grammarAccess.getParBTNodeAccess().getNodesChildNodeParserRuleCall_4_0());
            	    				
            	    pushFollow(FOLLOW_43);
            	    lv_nodes_6_0=ruleChildNode();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getParBTNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"nodes",
            	    						lv_nodes_6_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.ChildNode");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop36;
                }
            } while (true);

            otherlv_7=(Token)match(input,43,FOLLOW_2); 

            			newLeafNode(otherlv_7, grammarAccess.getParBTNodeAccess().getRightCurlyBracketKeyword_5());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleParBTNode"


    // $ANTLR start "entryRuleSeqBTNode"
    // InternalBTree.g:1916:1: entryRuleSeqBTNode returns [EObject current=null] : iv_ruleSeqBTNode= ruleSeqBTNode EOF ;
    public final EObject entryRuleSeqBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleSeqBTNode = null;


        try {
            // InternalBTree.g:1916:50: (iv_ruleSeqBTNode= ruleSeqBTNode EOF )
            // InternalBTree.g:1917:2: iv_ruleSeqBTNode= ruleSeqBTNode EOF
            {
             newCompositeNode(grammarAccess.getSeqBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleSeqBTNode=ruleSeqBTNode();

            state._fsp--;

             current =iv_ruleSeqBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleSeqBTNode"


    // $ANTLR start "ruleSeqBTNode"
    // InternalBTree.g:1923:1: ruleSeqBTNode returns [EObject current=null] : (otherlv_0= 'seq' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' ) ;
    public final EObject ruleSeqBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_4=null;
        Token otherlv_5=null;
        Token otherlv_7=null;
        AntlrDatatypeRuleToken lv_cond_3_0 = null;

        EObject lv_nodes_6_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:1929:2: ( (otherlv_0= 'seq' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' ) )
            // InternalBTree.g:1930:2: (otherlv_0= 'seq' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' )
            {
            // InternalBTree.g:1930:2: (otherlv_0= 'seq' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' )
            // InternalBTree.g:1931:3: otherlv_0= 'seq' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}'
            {
            otherlv_0=(Token)match(input,44,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getSeqBTNodeAccess().getSeqKeyword_0());
            		
            // InternalBTree.g:1935:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:1936:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:1936:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:1937:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_40); 

            					newLeafNode(lv_name_1_0, grammarAccess.getSeqBTNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getSeqBTNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:1953:3: (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )?
            int alt37=2;
            int LA37_0 = input.LA(1);

            if ( (LA37_0==15) ) {
                alt37=1;
            }
            switch (alt37) {
                case 1 :
                    // InternalBTree.g:1954:4: otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')'
                    {
                    otherlv_2=(Token)match(input,15,FOLLOW_41); 

                    				newLeafNode(otherlv_2, grammarAccess.getSeqBTNodeAccess().getLeftParenthesisKeyword_2_0());
                    			
                    // InternalBTree.g:1958:4: ( (lv_cond_3_0= ruleStatus ) )
                    // InternalBTree.g:1959:5: (lv_cond_3_0= ruleStatus )
                    {
                    // InternalBTree.g:1959:5: (lv_cond_3_0= ruleStatus )
                    // InternalBTree.g:1960:6: lv_cond_3_0= ruleStatus
                    {

                    						newCompositeNode(grammarAccess.getSeqBTNodeAccess().getCondStatusParserRuleCall_2_1_0());
                    					
                    pushFollow(FOLLOW_19);
                    lv_cond_3_0=ruleStatus();

                    state._fsp--;


                    						if (current==null) {
                    							current = createModelElementForParent(grammarAccess.getSeqBTNodeRule());
                    						}
                    						set(
                    							current,
                    							"cond",
                    							lv_cond_3_0,
                    							"edu.vanderbilt.isis.alc.btree.BTree.Status");
                    						afterParserOrEnumRuleCall();
                    					

                    }


                    }

                    otherlv_4=(Token)match(input,20,FOLLOW_42); 

                    				newLeafNode(otherlv_4, grammarAccess.getSeqBTNodeAccess().getRightParenthesisKeyword_2_2());
                    			

                    }
                    break;

            }

            otherlv_5=(Token)match(input,42,FOLLOW_43); 

            			newLeafNode(otherlv_5, grammarAccess.getSeqBTNodeAccess().getLeftCurlyBracketKeyword_3());
            		
            // InternalBTree.g:1986:3: ( (lv_nodes_6_0= ruleChildNode ) )*
            loop38:
            do {
                int alt38=2;
                int LA38_0 = input.LA(1);

                if ( (LA38_0==41||(LA38_0>=44 && LA38_0<=46)||(LA38_0>=49 && LA38_0<=52)) ) {
                    alt38=1;
                }


                switch (alt38) {
            	case 1 :
            	    // InternalBTree.g:1987:4: (lv_nodes_6_0= ruleChildNode )
            	    {
            	    // InternalBTree.g:1987:4: (lv_nodes_6_0= ruleChildNode )
            	    // InternalBTree.g:1988:5: lv_nodes_6_0= ruleChildNode
            	    {

            	    					newCompositeNode(grammarAccess.getSeqBTNodeAccess().getNodesChildNodeParserRuleCall_4_0());
            	    				
            	    pushFollow(FOLLOW_43);
            	    lv_nodes_6_0=ruleChildNode();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getSeqBTNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"nodes",
            	    						lv_nodes_6_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.ChildNode");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop38;
                }
            } while (true);

            otherlv_7=(Token)match(input,43,FOLLOW_2); 

            			newLeafNode(otherlv_7, grammarAccess.getSeqBTNodeAccess().getRightCurlyBracketKeyword_5());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleSeqBTNode"


    // $ANTLR start "entryRuleSelBTNode"
    // InternalBTree.g:2013:1: entryRuleSelBTNode returns [EObject current=null] : iv_ruleSelBTNode= ruleSelBTNode EOF ;
    public final EObject entryRuleSelBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleSelBTNode = null;


        try {
            // InternalBTree.g:2013:50: (iv_ruleSelBTNode= ruleSelBTNode EOF )
            // InternalBTree.g:2014:2: iv_ruleSelBTNode= ruleSelBTNode EOF
            {
             newCompositeNode(grammarAccess.getSelBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleSelBTNode=ruleSelBTNode();

            state._fsp--;

             current =iv_ruleSelBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleSelBTNode"


    // $ANTLR start "ruleSelBTNode"
    // InternalBTree.g:2020:1: ruleSelBTNode returns [EObject current=null] : (otherlv_0= 'sel' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' ) ;
    public final EObject ruleSelBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_4=null;
        Token otherlv_5=null;
        Token otherlv_7=null;
        AntlrDatatypeRuleToken lv_cond_3_0 = null;

        EObject lv_nodes_6_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:2026:2: ( (otherlv_0= 'sel' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' ) )
            // InternalBTree.g:2027:2: (otherlv_0= 'sel' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' )
            {
            // InternalBTree.g:2027:2: (otherlv_0= 'sel' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}' )
            // InternalBTree.g:2028:3: otherlv_0= 'sel' ( (lv_name_1_0= RULE_ID ) ) (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )? otherlv_5= '{' ( (lv_nodes_6_0= ruleChildNode ) )* otherlv_7= '}'
            {
            otherlv_0=(Token)match(input,45,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getSelBTNodeAccess().getSelKeyword_0());
            		
            // InternalBTree.g:2032:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:2033:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:2033:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:2034:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_40); 

            					newLeafNode(lv_name_1_0, grammarAccess.getSelBTNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getSelBTNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            // InternalBTree.g:2050:3: (otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')' )?
            int alt39=2;
            int LA39_0 = input.LA(1);

            if ( (LA39_0==15) ) {
                alt39=1;
            }
            switch (alt39) {
                case 1 :
                    // InternalBTree.g:2051:4: otherlv_2= '(' ( (lv_cond_3_0= ruleStatus ) ) otherlv_4= ')'
                    {
                    otherlv_2=(Token)match(input,15,FOLLOW_41); 

                    				newLeafNode(otherlv_2, grammarAccess.getSelBTNodeAccess().getLeftParenthesisKeyword_2_0());
                    			
                    // InternalBTree.g:2055:4: ( (lv_cond_3_0= ruleStatus ) )
                    // InternalBTree.g:2056:5: (lv_cond_3_0= ruleStatus )
                    {
                    // InternalBTree.g:2056:5: (lv_cond_3_0= ruleStatus )
                    // InternalBTree.g:2057:6: lv_cond_3_0= ruleStatus
                    {

                    						newCompositeNode(grammarAccess.getSelBTNodeAccess().getCondStatusParserRuleCall_2_1_0());
                    					
                    pushFollow(FOLLOW_19);
                    lv_cond_3_0=ruleStatus();

                    state._fsp--;


                    						if (current==null) {
                    							current = createModelElementForParent(grammarAccess.getSelBTNodeRule());
                    						}
                    						set(
                    							current,
                    							"cond",
                    							lv_cond_3_0,
                    							"edu.vanderbilt.isis.alc.btree.BTree.Status");
                    						afterParserOrEnumRuleCall();
                    					

                    }


                    }

                    otherlv_4=(Token)match(input,20,FOLLOW_42); 

                    				newLeafNode(otherlv_4, grammarAccess.getSelBTNodeAccess().getRightParenthesisKeyword_2_2());
                    			

                    }
                    break;

            }

            otherlv_5=(Token)match(input,42,FOLLOW_43); 

            			newLeafNode(otherlv_5, grammarAccess.getSelBTNodeAccess().getLeftCurlyBracketKeyword_3());
            		
            // InternalBTree.g:2083:3: ( (lv_nodes_6_0= ruleChildNode ) )*
            loop40:
            do {
                int alt40=2;
                int LA40_0 = input.LA(1);

                if ( (LA40_0==41||(LA40_0>=44 && LA40_0<=46)||(LA40_0>=49 && LA40_0<=52)) ) {
                    alt40=1;
                }


                switch (alt40) {
            	case 1 :
            	    // InternalBTree.g:2084:4: (lv_nodes_6_0= ruleChildNode )
            	    {
            	    // InternalBTree.g:2084:4: (lv_nodes_6_0= ruleChildNode )
            	    // InternalBTree.g:2085:5: lv_nodes_6_0= ruleChildNode
            	    {

            	    					newCompositeNode(grammarAccess.getSelBTNodeAccess().getNodesChildNodeParserRuleCall_4_0());
            	    				
            	    pushFollow(FOLLOW_43);
            	    lv_nodes_6_0=ruleChildNode();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getSelBTNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"nodes",
            	    						lv_nodes_6_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.ChildNode");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop40;
                }
            } while (true);

            otherlv_7=(Token)match(input,43,FOLLOW_2); 

            			newLeafNode(otherlv_7, grammarAccess.getSelBTNodeAccess().getRightCurlyBracketKeyword_5());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleSelBTNode"


    // $ANTLR start "entryRuleSIFBTNode"
    // InternalBTree.g:2110:1: entryRuleSIFBTNode returns [EObject current=null] : iv_ruleSIFBTNode= ruleSIFBTNode EOF ;
    public final EObject entryRuleSIFBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleSIFBTNode = null;


        try {
            // InternalBTree.g:2110:50: (iv_ruleSIFBTNode= ruleSIFBTNode EOF )
            // InternalBTree.g:2111:2: iv_ruleSIFBTNode= ruleSIFBTNode EOF
            {
             newCompositeNode(grammarAccess.getSIFBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleSIFBTNode=ruleSIFBTNode();

            state._fsp--;

             current =iv_ruleSIFBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleSIFBTNode"


    // $ANTLR start "ruleSIFBTNode"
    // InternalBTree.g:2117:1: ruleSIFBTNode returns [EObject current=null] : (otherlv_0= 'do' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '{' otherlv_3= 'if' ( (otherlv_4= RULE_ID ) ) (otherlv_5= ',' ( (otherlv_6= RULE_ID ) ) )* otherlv_7= 'then' otherlv_8= '{' ( (lv_nodes_9_0= ruleChildNode ) ) ( (lv_nodes_10_0= ruleChildNode ) )* otherlv_11= '}' otherlv_12= '}' ) ;
    public final EObject ruleSIFBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_3=null;
        Token otherlv_4=null;
        Token otherlv_5=null;
        Token otherlv_6=null;
        Token otherlv_7=null;
        Token otherlv_8=null;
        Token otherlv_11=null;
        Token otherlv_12=null;
        EObject lv_nodes_9_0 = null;

        EObject lv_nodes_10_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:2123:2: ( (otherlv_0= 'do' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '{' otherlv_3= 'if' ( (otherlv_4= RULE_ID ) ) (otherlv_5= ',' ( (otherlv_6= RULE_ID ) ) )* otherlv_7= 'then' otherlv_8= '{' ( (lv_nodes_9_0= ruleChildNode ) ) ( (lv_nodes_10_0= ruleChildNode ) )* otherlv_11= '}' otherlv_12= '}' ) )
            // InternalBTree.g:2124:2: (otherlv_0= 'do' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '{' otherlv_3= 'if' ( (otherlv_4= RULE_ID ) ) (otherlv_5= ',' ( (otherlv_6= RULE_ID ) ) )* otherlv_7= 'then' otherlv_8= '{' ( (lv_nodes_9_0= ruleChildNode ) ) ( (lv_nodes_10_0= ruleChildNode ) )* otherlv_11= '}' otherlv_12= '}' )
            {
            // InternalBTree.g:2124:2: (otherlv_0= 'do' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '{' otherlv_3= 'if' ( (otherlv_4= RULE_ID ) ) (otherlv_5= ',' ( (otherlv_6= RULE_ID ) ) )* otherlv_7= 'then' otherlv_8= '{' ( (lv_nodes_9_0= ruleChildNode ) ) ( (lv_nodes_10_0= ruleChildNode ) )* otherlv_11= '}' otherlv_12= '}' )
            // InternalBTree.g:2125:3: otherlv_0= 'do' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '{' otherlv_3= 'if' ( (otherlv_4= RULE_ID ) ) (otherlv_5= ',' ( (otherlv_6= RULE_ID ) ) )* otherlv_7= 'then' otherlv_8= '{' ( (lv_nodes_9_0= ruleChildNode ) ) ( (lv_nodes_10_0= ruleChildNode ) )* otherlv_11= '}' otherlv_12= '}'
            {
            otherlv_0=(Token)match(input,46,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getSIFBTNodeAccess().getDoKeyword_0());
            		
            // InternalBTree.g:2129:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:2130:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:2130:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:2131:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_42); 

            					newLeafNode(lv_name_1_0, grammarAccess.getSIFBTNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getSIFBTNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            otherlv_2=(Token)match(input,42,FOLLOW_44); 

            			newLeafNode(otherlv_2, grammarAccess.getSIFBTNodeAccess().getLeftCurlyBracketKeyword_2());
            		
            otherlv_3=(Token)match(input,47,FOLLOW_3); 

            			newLeafNode(otherlv_3, grammarAccess.getSIFBTNodeAccess().getIfKeyword_3());
            		
            // InternalBTree.g:2155:3: ( (otherlv_4= RULE_ID ) )
            // InternalBTree.g:2156:4: (otherlv_4= RULE_ID )
            {
            // InternalBTree.g:2156:4: (otherlv_4= RULE_ID )
            // InternalBTree.g:2157:5: otherlv_4= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getSIFBTNodeRule());
            					}
            				
            otherlv_4=(Token)match(input,RULE_ID,FOLLOW_45); 

            					newLeafNode(otherlv_4, grammarAccess.getSIFBTNodeAccess().getChecksCheckNodeCrossReference_4_0());
            				

            }


            }

            // InternalBTree.g:2168:3: (otherlv_5= ',' ( (otherlv_6= RULE_ID ) ) )*
            loop41:
            do {
                int alt41=2;
                int LA41_0 = input.LA(1);

                if ( (LA41_0==18) ) {
                    alt41=1;
                }


                switch (alt41) {
            	case 1 :
            	    // InternalBTree.g:2169:4: otherlv_5= ',' ( (otherlv_6= RULE_ID ) )
            	    {
            	    otherlv_5=(Token)match(input,18,FOLLOW_3); 

            	    				newLeafNode(otherlv_5, grammarAccess.getSIFBTNodeAccess().getCommaKeyword_5_0());
            	    			
            	    // InternalBTree.g:2173:4: ( (otherlv_6= RULE_ID ) )
            	    // InternalBTree.g:2174:5: (otherlv_6= RULE_ID )
            	    {
            	    // InternalBTree.g:2174:5: (otherlv_6= RULE_ID )
            	    // InternalBTree.g:2175:6: otherlv_6= RULE_ID
            	    {

            	    						if (current==null) {
            	    							current = createModelElement(grammarAccess.getSIFBTNodeRule());
            	    						}
            	    					
            	    otherlv_6=(Token)match(input,RULE_ID,FOLLOW_45); 

            	    						newLeafNode(otherlv_6, grammarAccess.getSIFBTNodeAccess().getChecksCheckNodeCrossReference_5_1_0());
            	    					

            	    }


            	    }


            	    }
            	    break;

            	default :
            	    break loop41;
                }
            } while (true);

            otherlv_7=(Token)match(input,48,FOLLOW_42); 

            			newLeafNode(otherlv_7, grammarAccess.getSIFBTNodeAccess().getThenKeyword_6());
            		
            otherlv_8=(Token)match(input,42,FOLLOW_20); 

            			newLeafNode(otherlv_8, grammarAccess.getSIFBTNodeAccess().getLeftCurlyBracketKeyword_7());
            		
            // InternalBTree.g:2195:3: ( (lv_nodes_9_0= ruleChildNode ) )
            // InternalBTree.g:2196:4: (lv_nodes_9_0= ruleChildNode )
            {
            // InternalBTree.g:2196:4: (lv_nodes_9_0= ruleChildNode )
            // InternalBTree.g:2197:5: lv_nodes_9_0= ruleChildNode
            {

            					newCompositeNode(grammarAccess.getSIFBTNodeAccess().getNodesChildNodeParserRuleCall_8_0());
            				
            pushFollow(FOLLOW_43);
            lv_nodes_9_0=ruleChildNode();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getSIFBTNodeRule());
            					}
            					add(
            						current,
            						"nodes",
            						lv_nodes_9_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.ChildNode");
            					afterParserOrEnumRuleCall();
            				

            }


            }

            // InternalBTree.g:2214:3: ( (lv_nodes_10_0= ruleChildNode ) )*
            loop42:
            do {
                int alt42=2;
                int LA42_0 = input.LA(1);

                if ( (LA42_0==41||(LA42_0>=44 && LA42_0<=46)||(LA42_0>=49 && LA42_0<=52)) ) {
                    alt42=1;
                }


                switch (alt42) {
            	case 1 :
            	    // InternalBTree.g:2215:4: (lv_nodes_10_0= ruleChildNode )
            	    {
            	    // InternalBTree.g:2215:4: (lv_nodes_10_0= ruleChildNode )
            	    // InternalBTree.g:2216:5: lv_nodes_10_0= ruleChildNode
            	    {

            	    					newCompositeNode(grammarAccess.getSIFBTNodeAccess().getNodesChildNodeParserRuleCall_9_0());
            	    				
            	    pushFollow(FOLLOW_43);
            	    lv_nodes_10_0=ruleChildNode();

            	    state._fsp--;


            	    					if (current==null) {
            	    						current = createModelElementForParent(grammarAccess.getSIFBTNodeRule());
            	    					}
            	    					add(
            	    						current,
            	    						"nodes",
            	    						lv_nodes_10_0,
            	    						"edu.vanderbilt.isis.alc.btree.BTree.ChildNode");
            	    					afterParserOrEnumRuleCall();
            	    				

            	    }


            	    }
            	    break;

            	default :
            	    break loop42;
                }
            } while (true);

            otherlv_11=(Token)match(input,43,FOLLOW_46); 

            			newLeafNode(otherlv_11, grammarAccess.getSIFBTNodeAccess().getRightCurlyBracketKeyword_10());
            		
            otherlv_12=(Token)match(input,43,FOLLOW_2); 

            			newLeafNode(otherlv_12, grammarAccess.getSIFBTNodeAccess().getRightCurlyBracketKeyword_11());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleSIFBTNode"


    // $ANTLR start "entryRuleMonBTNode"
    // InternalBTree.g:2245:1: entryRuleMonBTNode returns [EObject current=null] : iv_ruleMonBTNode= ruleMonBTNode EOF ;
    public final EObject entryRuleMonBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleMonBTNode = null;


        try {
            // InternalBTree.g:2245:50: (iv_ruleMonBTNode= ruleMonBTNode EOF )
            // InternalBTree.g:2246:2: iv_ruleMonBTNode= ruleMonBTNode EOF
            {
             newCompositeNode(grammarAccess.getMonBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleMonBTNode=ruleMonBTNode();

            state._fsp--;

             current =iv_ruleMonBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleMonBTNode"


    // $ANTLR start "ruleMonBTNode"
    // InternalBTree.g:2252:1: ruleMonBTNode returns [EObject current=null] : (otherlv_0= 'mon' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* ) ;
    public final EObject ruleMonBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token otherlv_1=null;
        Token otherlv_2=null;
        Token otherlv_3=null;


        	enterRule();

        try {
            // InternalBTree.g:2258:2: ( (otherlv_0= 'mon' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* ) )
            // InternalBTree.g:2259:2: (otherlv_0= 'mon' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* )
            {
            // InternalBTree.g:2259:2: (otherlv_0= 'mon' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* )
            // InternalBTree.g:2260:3: otherlv_0= 'mon' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )*
            {
            otherlv_0=(Token)match(input,49,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getMonBTNodeAccess().getMonKeyword_0());
            		
            // InternalBTree.g:2264:3: ( (otherlv_1= RULE_ID ) )
            // InternalBTree.g:2265:4: (otherlv_1= RULE_ID )
            {
            // InternalBTree.g:2265:4: (otherlv_1= RULE_ID )
            // InternalBTree.g:2266:5: otherlv_1= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getMonBTNodeRule());
            					}
            				
            otherlv_1=(Token)match(input,RULE_ID,FOLLOW_47); 

            					newLeafNode(otherlv_1, grammarAccess.getMonBTNodeAccess().getMonBBNodeCrossReference_1_0());
            				

            }


            }

            // InternalBTree.g:2277:3: (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )*
            loop43:
            do {
                int alt43=2;
                int LA43_0 = input.LA(1);

                if ( (LA43_0==18) ) {
                    alt43=1;
                }


                switch (alt43) {
            	case 1 :
            	    // InternalBTree.g:2278:4: otherlv_2= ',' ( (otherlv_3= RULE_ID ) )
            	    {
            	    otherlv_2=(Token)match(input,18,FOLLOW_3); 

            	    				newLeafNode(otherlv_2, grammarAccess.getMonBTNodeAccess().getCommaKeyword_2_0());
            	    			
            	    // InternalBTree.g:2282:4: ( (otherlv_3= RULE_ID ) )
            	    // InternalBTree.g:2283:5: (otherlv_3= RULE_ID )
            	    {
            	    // InternalBTree.g:2283:5: (otherlv_3= RULE_ID )
            	    // InternalBTree.g:2284:6: otherlv_3= RULE_ID
            	    {

            	    						if (current==null) {
            	    							current = createModelElement(grammarAccess.getMonBTNodeRule());
            	    						}
            	    					
            	    otherlv_3=(Token)match(input,RULE_ID,FOLLOW_47); 

            	    						newLeafNode(otherlv_3, grammarAccess.getMonBTNodeAccess().getMonBBNodeCrossReference_2_1_0());
            	    					

            	    }


            	    }


            	    }
            	    break;

            	default :
            	    break loop43;
                }
            } while (true);


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleMonBTNode"


    // $ANTLR start "entryRuleTaskBTNode"
    // InternalBTree.g:2300:1: entryRuleTaskBTNode returns [EObject current=null] : iv_ruleTaskBTNode= ruleTaskBTNode EOF ;
    public final EObject entryRuleTaskBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleTaskBTNode = null;


        try {
            // InternalBTree.g:2300:51: (iv_ruleTaskBTNode= ruleTaskBTNode EOF )
            // InternalBTree.g:2301:2: iv_ruleTaskBTNode= ruleTaskBTNode EOF
            {
             newCompositeNode(grammarAccess.getTaskBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleTaskBTNode=ruleTaskBTNode();

            state._fsp--;

             current =iv_ruleTaskBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleTaskBTNode"


    // $ANTLR start "ruleTaskBTNode"
    // InternalBTree.g:2307:1: ruleTaskBTNode returns [EObject current=null] : (otherlv_0= 'exec' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* ) ;
    public final EObject ruleTaskBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token otherlv_1=null;
        Token otherlv_2=null;
        Token otherlv_3=null;


        	enterRule();

        try {
            // InternalBTree.g:2313:2: ( (otherlv_0= 'exec' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* ) )
            // InternalBTree.g:2314:2: (otherlv_0= 'exec' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* )
            {
            // InternalBTree.g:2314:2: (otherlv_0= 'exec' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* )
            // InternalBTree.g:2315:3: otherlv_0= 'exec' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )*
            {
            otherlv_0=(Token)match(input,50,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getTaskBTNodeAccess().getExecKeyword_0());
            		
            // InternalBTree.g:2319:3: ( (otherlv_1= RULE_ID ) )
            // InternalBTree.g:2320:4: (otherlv_1= RULE_ID )
            {
            // InternalBTree.g:2320:4: (otherlv_1= RULE_ID )
            // InternalBTree.g:2321:5: otherlv_1= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTaskBTNodeRule());
            					}
            				
            otherlv_1=(Token)match(input,RULE_ID,FOLLOW_47); 

            					newLeafNode(otherlv_1, grammarAccess.getTaskBTNodeAccess().getTaskBehaviorNodeCrossReference_1_0());
            				

            }


            }

            // InternalBTree.g:2332:3: (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )*
            loop44:
            do {
                int alt44=2;
                int LA44_0 = input.LA(1);

                if ( (LA44_0==18) ) {
                    alt44=1;
                }


                switch (alt44) {
            	case 1 :
            	    // InternalBTree.g:2333:4: otherlv_2= ',' ( (otherlv_3= RULE_ID ) )
            	    {
            	    otherlv_2=(Token)match(input,18,FOLLOW_3); 

            	    				newLeafNode(otherlv_2, grammarAccess.getTaskBTNodeAccess().getCommaKeyword_2_0());
            	    			
            	    // InternalBTree.g:2337:4: ( (otherlv_3= RULE_ID ) )
            	    // InternalBTree.g:2338:5: (otherlv_3= RULE_ID )
            	    {
            	    // InternalBTree.g:2338:5: (otherlv_3= RULE_ID )
            	    // InternalBTree.g:2339:6: otherlv_3= RULE_ID
            	    {

            	    						if (current==null) {
            	    							current = createModelElement(grammarAccess.getTaskBTNodeRule());
            	    						}
            	    					
            	    otherlv_3=(Token)match(input,RULE_ID,FOLLOW_47); 

            	    						newLeafNode(otherlv_3, grammarAccess.getTaskBTNodeAccess().getTaskBehaviorNodeCrossReference_2_1_0());
            	    					

            	    }


            	    }


            	    }
            	    break;

            	default :
            	    break loop44;
                }
            } while (true);


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleTaskBTNode"


    // $ANTLR start "entryRuleTimerBTNode"
    // InternalBTree.g:2355:1: entryRuleTimerBTNode returns [EObject current=null] : iv_ruleTimerBTNode= ruleTimerBTNode EOF ;
    public final EObject entryRuleTimerBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleTimerBTNode = null;


        try {
            // InternalBTree.g:2355:52: (iv_ruleTimerBTNode= ruleTimerBTNode EOF )
            // InternalBTree.g:2356:2: iv_ruleTimerBTNode= ruleTimerBTNode EOF
            {
             newCompositeNode(grammarAccess.getTimerBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleTimerBTNode=ruleTimerBTNode();

            state._fsp--;

             current =iv_ruleTimerBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleTimerBTNode"


    // $ANTLR start "ruleTimerBTNode"
    // InternalBTree.g:2362:1: ruleTimerBTNode returns [EObject current=null] : (otherlv_0= 'timer' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '(' ( (lv_duration_3_0= ruleNUMBER ) ) otherlv_4= ')' ) ;
    public final EObject ruleTimerBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token lv_name_1_0=null;
        Token otherlv_2=null;
        Token otherlv_4=null;
        AntlrDatatypeRuleToken lv_duration_3_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:2368:2: ( (otherlv_0= 'timer' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '(' ( (lv_duration_3_0= ruleNUMBER ) ) otherlv_4= ')' ) )
            // InternalBTree.g:2369:2: (otherlv_0= 'timer' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '(' ( (lv_duration_3_0= ruleNUMBER ) ) otherlv_4= ')' )
            {
            // InternalBTree.g:2369:2: (otherlv_0= 'timer' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '(' ( (lv_duration_3_0= ruleNUMBER ) ) otherlv_4= ')' )
            // InternalBTree.g:2370:3: otherlv_0= 'timer' ( (lv_name_1_0= RULE_ID ) ) otherlv_2= '(' ( (lv_duration_3_0= ruleNUMBER ) ) otherlv_4= ')'
            {
            otherlv_0=(Token)match(input,51,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getTimerBTNodeAccess().getTimerKeyword_0());
            		
            // InternalBTree.g:2374:3: ( (lv_name_1_0= RULE_ID ) )
            // InternalBTree.g:2375:4: (lv_name_1_0= RULE_ID )
            {
            // InternalBTree.g:2375:4: (lv_name_1_0= RULE_ID )
            // InternalBTree.g:2376:5: lv_name_1_0= RULE_ID
            {
            lv_name_1_0=(Token)match(input,RULE_ID,FOLLOW_13); 

            					newLeafNode(lv_name_1_0, grammarAccess.getTimerBTNodeAccess().getNameIDTerminalRuleCall_1_0());
            				

            					if (current==null) {
            						current = createModelElement(grammarAccess.getTimerBTNodeRule());
            					}
            					setWithLastConsumed(
            						current,
            						"name",
            						lv_name_1_0,
            						"org.eclipse.xtext.common.Terminals.ID");
            				

            }


            }

            otherlv_2=(Token)match(input,15,FOLLOW_16); 

            			newLeafNode(otherlv_2, grammarAccess.getTimerBTNodeAccess().getLeftParenthesisKeyword_2());
            		
            // InternalBTree.g:2396:3: ( (lv_duration_3_0= ruleNUMBER ) )
            // InternalBTree.g:2397:4: (lv_duration_3_0= ruleNUMBER )
            {
            // InternalBTree.g:2397:4: (lv_duration_3_0= ruleNUMBER )
            // InternalBTree.g:2398:5: lv_duration_3_0= ruleNUMBER
            {

            					newCompositeNode(grammarAccess.getTimerBTNodeAccess().getDurationNUMBERParserRuleCall_3_0());
            				
            pushFollow(FOLLOW_19);
            lv_duration_3_0=ruleNUMBER();

            state._fsp--;


            					if (current==null) {
            						current = createModelElementForParent(grammarAccess.getTimerBTNodeRule());
            					}
            					set(
            						current,
            						"duration",
            						lv_duration_3_0,
            						"edu.vanderbilt.isis.alc.btree.BTree.NUMBER");
            					afterParserOrEnumRuleCall();
            				

            }


            }

            otherlv_4=(Token)match(input,20,FOLLOW_2); 

            			newLeafNode(otherlv_4, grammarAccess.getTimerBTNodeAccess().getRightParenthesisKeyword_4());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleTimerBTNode"


    // $ANTLR start "entryRuleCheckBTNode"
    // InternalBTree.g:2423:1: entryRuleCheckBTNode returns [EObject current=null] : iv_ruleCheckBTNode= ruleCheckBTNode EOF ;
    public final EObject entryRuleCheckBTNode() throws RecognitionException {
        EObject current = null;

        EObject iv_ruleCheckBTNode = null;


        try {
            // InternalBTree.g:2423:52: (iv_ruleCheckBTNode= ruleCheckBTNode EOF )
            // InternalBTree.g:2424:2: iv_ruleCheckBTNode= ruleCheckBTNode EOF
            {
             newCompositeNode(grammarAccess.getCheckBTNodeRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleCheckBTNode=ruleCheckBTNode();

            state._fsp--;

             current =iv_ruleCheckBTNode; 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleCheckBTNode"


    // $ANTLR start "ruleCheckBTNode"
    // InternalBTree.g:2430:1: ruleCheckBTNode returns [EObject current=null] : (otherlv_0= 'chk' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* ) ;
    public final EObject ruleCheckBTNode() throws RecognitionException {
        EObject current = null;

        Token otherlv_0=null;
        Token otherlv_1=null;
        Token otherlv_2=null;
        Token otherlv_3=null;


        	enterRule();

        try {
            // InternalBTree.g:2436:2: ( (otherlv_0= 'chk' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* ) )
            // InternalBTree.g:2437:2: (otherlv_0= 'chk' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* )
            {
            // InternalBTree.g:2437:2: (otherlv_0= 'chk' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )* )
            // InternalBTree.g:2438:3: otherlv_0= 'chk' ( (otherlv_1= RULE_ID ) ) (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )*
            {
            otherlv_0=(Token)match(input,52,FOLLOW_3); 

            			newLeafNode(otherlv_0, grammarAccess.getCheckBTNodeAccess().getChkKeyword_0());
            		
            // InternalBTree.g:2442:3: ( (otherlv_1= RULE_ID ) )
            // InternalBTree.g:2443:4: (otherlv_1= RULE_ID )
            {
            // InternalBTree.g:2443:4: (otherlv_1= RULE_ID )
            // InternalBTree.g:2444:5: otherlv_1= RULE_ID
            {

            					if (current==null) {
            						current = createModelElement(grammarAccess.getCheckBTNodeRule());
            					}
            				
            otherlv_1=(Token)match(input,RULE_ID,FOLLOW_47); 

            					newLeafNode(otherlv_1, grammarAccess.getCheckBTNodeAccess().getCheckCheckNodeCrossReference_1_0());
            				

            }


            }

            // InternalBTree.g:2455:3: (otherlv_2= ',' ( (otherlv_3= RULE_ID ) ) )*
            loop45:
            do {
                int alt45=2;
                int LA45_0 = input.LA(1);

                if ( (LA45_0==18) ) {
                    alt45=1;
                }


                switch (alt45) {
            	case 1 :
            	    // InternalBTree.g:2456:4: otherlv_2= ',' ( (otherlv_3= RULE_ID ) )
            	    {
            	    otherlv_2=(Token)match(input,18,FOLLOW_3); 

            	    				newLeafNode(otherlv_2, grammarAccess.getCheckBTNodeAccess().getCommaKeyword_2_0());
            	    			
            	    // InternalBTree.g:2460:4: ( (otherlv_3= RULE_ID ) )
            	    // InternalBTree.g:2461:5: (otherlv_3= RULE_ID )
            	    {
            	    // InternalBTree.g:2461:5: (otherlv_3= RULE_ID )
            	    // InternalBTree.g:2462:6: otherlv_3= RULE_ID
            	    {

            	    						if (current==null) {
            	    							current = createModelElement(grammarAccess.getCheckBTNodeRule());
            	    						}
            	    					
            	    otherlv_3=(Token)match(input,RULE_ID,FOLLOW_47); 

            	    						newLeafNode(otherlv_3, grammarAccess.getCheckBTNodeAccess().getCheckCheckNodeCrossReference_2_1_0());
            	    					

            	    }


            	    }


            	    }
            	    break;

            	default :
            	    break loop45;
                }
            } while (true);


            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleCheckBTNode"


    // $ANTLR start "entryRuleStatus"
    // InternalBTree.g:2478:1: entryRuleStatus returns [String current=null] : iv_ruleStatus= ruleStatus EOF ;
    public final String entryRuleStatus() throws RecognitionException {
        String current = null;

        AntlrDatatypeRuleToken iv_ruleStatus = null;


        try {
            // InternalBTree.g:2478:46: (iv_ruleStatus= ruleStatus EOF )
            // InternalBTree.g:2479:2: iv_ruleStatus= ruleStatus EOF
            {
             newCompositeNode(grammarAccess.getStatusRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleStatus=ruleStatus();

            state._fsp--;

             current =iv_ruleStatus.getText(); 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleStatus"


    // $ANTLR start "ruleStatus"
    // InternalBTree.g:2485:1: ruleStatus returns [AntlrDatatypeRuleToken current=new AntlrDatatypeRuleToken()] : (kw= 'success' | kw= 'failure' | kw= 'running' | kw= 'invalid' ) ;
    public final AntlrDatatypeRuleToken ruleStatus() throws RecognitionException {
        AntlrDatatypeRuleToken current = new AntlrDatatypeRuleToken();

        Token kw=null;


        	enterRule();

        try {
            // InternalBTree.g:2491:2: ( (kw= 'success' | kw= 'failure' | kw= 'running' | kw= 'invalid' ) )
            // InternalBTree.g:2492:2: (kw= 'success' | kw= 'failure' | kw= 'running' | kw= 'invalid' )
            {
            // InternalBTree.g:2492:2: (kw= 'success' | kw= 'failure' | kw= 'running' | kw= 'invalid' )
            int alt46=4;
            switch ( input.LA(1) ) {
            case 35:
                {
                alt46=1;
                }
                break;
            case 36:
                {
                alt46=2;
                }
                break;
            case 37:
                {
                alt46=3;
                }
                break;
            case 53:
                {
                alt46=4;
                }
                break;
            default:
                NoViableAltException nvae =
                    new NoViableAltException("", 46, 0, input);

                throw nvae;
            }

            switch (alt46) {
                case 1 :
                    // InternalBTree.g:2493:3: kw= 'success'
                    {
                    kw=(Token)match(input,35,FOLLOW_2); 

                    			current.merge(kw);
                    			newLeafNode(kw, grammarAccess.getStatusAccess().getSuccessKeyword_0());
                    		

                    }
                    break;
                case 2 :
                    // InternalBTree.g:2499:3: kw= 'failure'
                    {
                    kw=(Token)match(input,36,FOLLOW_2); 

                    			current.merge(kw);
                    			newLeafNode(kw, grammarAccess.getStatusAccess().getFailureKeyword_1());
                    		

                    }
                    break;
                case 3 :
                    // InternalBTree.g:2505:3: kw= 'running'
                    {
                    kw=(Token)match(input,37,FOLLOW_2); 

                    			current.merge(kw);
                    			newLeafNode(kw, grammarAccess.getStatusAccess().getRunningKeyword_2());
                    		

                    }
                    break;
                case 4 :
                    // InternalBTree.g:2511:3: kw= 'invalid'
                    {
                    kw=(Token)match(input,53,FOLLOW_2); 

                    			current.merge(kw);
                    			newLeafNode(kw, grammarAccess.getStatusAccess().getInvalidKeyword_3());
                    		

                    }
                    break;

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleStatus"


    // $ANTLR start "entryRuleFLOAT"
    // InternalBTree.g:2520:1: entryRuleFLOAT returns [String current=null] : iv_ruleFLOAT= ruleFLOAT EOF ;
    public final String entryRuleFLOAT() throws RecognitionException {
        String current = null;

        AntlrDatatypeRuleToken iv_ruleFLOAT = null;


        try {
            // InternalBTree.g:2520:45: (iv_ruleFLOAT= ruleFLOAT EOF )
            // InternalBTree.g:2521:2: iv_ruleFLOAT= ruleFLOAT EOF
            {
             newCompositeNode(grammarAccess.getFLOATRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleFLOAT=ruleFLOAT();

            state._fsp--;

             current =iv_ruleFLOAT.getText(); 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleFLOAT"


    // $ANTLR start "ruleFLOAT"
    // InternalBTree.g:2527:1: ruleFLOAT returns [AntlrDatatypeRuleToken current=new AntlrDatatypeRuleToken()] : ( (kw= '-' )? this_INT_1= RULE_INT kw= '.' this_INT_3= RULE_INT ) ;
    public final AntlrDatatypeRuleToken ruleFLOAT() throws RecognitionException {
        AntlrDatatypeRuleToken current = new AntlrDatatypeRuleToken();

        Token kw=null;
        Token this_INT_1=null;
        Token this_INT_3=null;


        	enterRule();

        try {
            // InternalBTree.g:2533:2: ( ( (kw= '-' )? this_INT_1= RULE_INT kw= '.' this_INT_3= RULE_INT ) )
            // InternalBTree.g:2534:2: ( (kw= '-' )? this_INT_1= RULE_INT kw= '.' this_INT_3= RULE_INT )
            {
            // InternalBTree.g:2534:2: ( (kw= '-' )? this_INT_1= RULE_INT kw= '.' this_INT_3= RULE_INT )
            // InternalBTree.g:2535:3: (kw= '-' )? this_INT_1= RULE_INT kw= '.' this_INT_3= RULE_INT
            {
            // InternalBTree.g:2535:3: (kw= '-' )?
            int alt47=2;
            int LA47_0 = input.LA(1);

            if ( (LA47_0==54) ) {
                alt47=1;
            }
            switch (alt47) {
                case 1 :
                    // InternalBTree.g:2536:4: kw= '-'
                    {
                    kw=(Token)match(input,54,FOLLOW_48); 

                    				current.merge(kw);
                    				newLeafNode(kw, grammarAccess.getFLOATAccess().getHyphenMinusKeyword_0());
                    			

                    }
                    break;

            }

            this_INT_1=(Token)match(input,RULE_INT,FOLLOW_49); 

            			current.merge(this_INT_1);
            		

            			newLeafNode(this_INT_1, grammarAccess.getFLOATAccess().getINTTerminalRuleCall_1());
            		
            kw=(Token)match(input,55,FOLLOW_48); 

            			current.merge(kw);
            			newLeafNode(kw, grammarAccess.getFLOATAccess().getFullStopKeyword_2());
            		
            this_INT_3=(Token)match(input,RULE_INT,FOLLOW_2); 

            			current.merge(this_INT_3);
            		

            			newLeafNode(this_INT_3, grammarAccess.getFLOATAccess().getINTTerminalRuleCall_3());
            		

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleFLOAT"


    // $ANTLR start "entryRuleBASETYPE"
    // InternalBTree.g:2565:1: entryRuleBASETYPE returns [String current=null] : iv_ruleBASETYPE= ruleBASETYPE EOF ;
    public final String entryRuleBASETYPE() throws RecognitionException {
        String current = null;

        AntlrDatatypeRuleToken iv_ruleBASETYPE = null;


        try {
            // InternalBTree.g:2565:48: (iv_ruleBASETYPE= ruleBASETYPE EOF )
            // InternalBTree.g:2566:2: iv_ruleBASETYPE= ruleBASETYPE EOF
            {
             newCompositeNode(grammarAccess.getBASETYPERule()); 
            pushFollow(FOLLOW_1);
            iv_ruleBASETYPE=ruleBASETYPE();

            state._fsp--;

             current =iv_ruleBASETYPE.getText(); 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleBASETYPE"


    // $ANTLR start "ruleBASETYPE"
    // InternalBTree.g:2572:1: ruleBASETYPE returns [AntlrDatatypeRuleToken current=new AntlrDatatypeRuleToken()] : (this_STRING_0= RULE_STRING | this_FLOAT_1= ruleFLOAT | this_INT_2= RULE_INT | this_BOOLEAN_3= RULE_BOOLEAN ) ;
    public final AntlrDatatypeRuleToken ruleBASETYPE() throws RecognitionException {
        AntlrDatatypeRuleToken current = new AntlrDatatypeRuleToken();

        Token this_STRING_0=null;
        Token this_INT_2=null;
        Token this_BOOLEAN_3=null;
        AntlrDatatypeRuleToken this_FLOAT_1 = null;



        	enterRule();

        try {
            // InternalBTree.g:2578:2: ( (this_STRING_0= RULE_STRING | this_FLOAT_1= ruleFLOAT | this_INT_2= RULE_INT | this_BOOLEAN_3= RULE_BOOLEAN ) )
            // InternalBTree.g:2579:2: (this_STRING_0= RULE_STRING | this_FLOAT_1= ruleFLOAT | this_INT_2= RULE_INT | this_BOOLEAN_3= RULE_BOOLEAN )
            {
            // InternalBTree.g:2579:2: (this_STRING_0= RULE_STRING | this_FLOAT_1= ruleFLOAT | this_INT_2= RULE_INT | this_BOOLEAN_3= RULE_BOOLEAN )
            int alt48=4;
            switch ( input.LA(1) ) {
            case RULE_STRING:
                {
                alt48=1;
                }
                break;
            case 54:
                {
                alt48=2;
                }
                break;
            case RULE_INT:
                {
                int LA48_3 = input.LA(2);

                if ( (LA48_3==EOF||LA48_3==13||LA48_3==18||LA48_3==25) ) {
                    alt48=3;
                }
                else if ( (LA48_3==55) ) {
                    alt48=2;
                }
                else {
                    NoViableAltException nvae =
                        new NoViableAltException("", 48, 3, input);

                    throw nvae;
                }
                }
                break;
            case RULE_BOOLEAN:
                {
                alt48=4;
                }
                break;
            default:
                NoViableAltException nvae =
                    new NoViableAltException("", 48, 0, input);

                throw nvae;
            }

            switch (alt48) {
                case 1 :
                    // InternalBTree.g:2580:3: this_STRING_0= RULE_STRING
                    {
                    this_STRING_0=(Token)match(input,RULE_STRING,FOLLOW_2); 

                    			current.merge(this_STRING_0);
                    		

                    			newLeafNode(this_STRING_0, grammarAccess.getBASETYPEAccess().getSTRINGTerminalRuleCall_0());
                    		

                    }
                    break;
                case 2 :
                    // InternalBTree.g:2588:3: this_FLOAT_1= ruleFLOAT
                    {

                    			newCompositeNode(grammarAccess.getBASETYPEAccess().getFLOATParserRuleCall_1());
                    		
                    pushFollow(FOLLOW_2);
                    this_FLOAT_1=ruleFLOAT();

                    state._fsp--;


                    			current.merge(this_FLOAT_1);
                    		

                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 3 :
                    // InternalBTree.g:2599:3: this_INT_2= RULE_INT
                    {
                    this_INT_2=(Token)match(input,RULE_INT,FOLLOW_2); 

                    			current.merge(this_INT_2);
                    		

                    			newLeafNode(this_INT_2, grammarAccess.getBASETYPEAccess().getINTTerminalRuleCall_2());
                    		

                    }
                    break;
                case 4 :
                    // InternalBTree.g:2607:3: this_BOOLEAN_3= RULE_BOOLEAN
                    {
                    this_BOOLEAN_3=(Token)match(input,RULE_BOOLEAN,FOLLOW_2); 

                    			current.merge(this_BOOLEAN_3);
                    		

                    			newLeafNode(this_BOOLEAN_3, grammarAccess.getBASETYPEAccess().getBOOLEANTerminalRuleCall_3());
                    		

                    }
                    break;

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleBASETYPE"


    // $ANTLR start "entryRuleNUMBER"
    // InternalBTree.g:2618:1: entryRuleNUMBER returns [String current=null] : iv_ruleNUMBER= ruleNUMBER EOF ;
    public final String entryRuleNUMBER() throws RecognitionException {
        String current = null;

        AntlrDatatypeRuleToken iv_ruleNUMBER = null;


        try {
            // InternalBTree.g:2618:46: (iv_ruleNUMBER= ruleNUMBER EOF )
            // InternalBTree.g:2619:2: iv_ruleNUMBER= ruleNUMBER EOF
            {
             newCompositeNode(grammarAccess.getNUMBERRule()); 
            pushFollow(FOLLOW_1);
            iv_ruleNUMBER=ruleNUMBER();

            state._fsp--;

             current =iv_ruleNUMBER.getText(); 
            match(input,EOF,FOLLOW_2); 

            }

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "entryRuleNUMBER"


    // $ANTLR start "ruleNUMBER"
    // InternalBTree.g:2625:1: ruleNUMBER returns [AntlrDatatypeRuleToken current=new AntlrDatatypeRuleToken()] : (this_FLOAT_0= ruleFLOAT | this_INT_1= RULE_INT ) ;
    public final AntlrDatatypeRuleToken ruleNUMBER() throws RecognitionException {
        AntlrDatatypeRuleToken current = new AntlrDatatypeRuleToken();

        Token this_INT_1=null;
        AntlrDatatypeRuleToken this_FLOAT_0 = null;



        	enterRule();

        try {
            // InternalBTree.g:2631:2: ( (this_FLOAT_0= ruleFLOAT | this_INT_1= RULE_INT ) )
            // InternalBTree.g:2632:2: (this_FLOAT_0= ruleFLOAT | this_INT_1= RULE_INT )
            {
            // InternalBTree.g:2632:2: (this_FLOAT_0= ruleFLOAT | this_INT_1= RULE_INT )
            int alt49=2;
            int LA49_0 = input.LA(1);

            if ( (LA49_0==54) ) {
                alt49=1;
            }
            else if ( (LA49_0==RULE_INT) ) {
                int LA49_2 = input.LA(2);

                if ( (LA49_2==55) ) {
                    alt49=1;
                }
                else if ( (LA49_2==EOF||LA49_2==20) ) {
                    alt49=2;
                }
                else {
                    NoViableAltException nvae =
                        new NoViableAltException("", 49, 2, input);

                    throw nvae;
                }
            }
            else {
                NoViableAltException nvae =
                    new NoViableAltException("", 49, 0, input);

                throw nvae;
            }
            switch (alt49) {
                case 1 :
                    // InternalBTree.g:2633:3: this_FLOAT_0= ruleFLOAT
                    {

                    			newCompositeNode(grammarAccess.getNUMBERAccess().getFLOATParserRuleCall_0());
                    		
                    pushFollow(FOLLOW_2);
                    this_FLOAT_0=ruleFLOAT();

                    state._fsp--;


                    			current.merge(this_FLOAT_0);
                    		

                    			afterParserOrEnumRuleCall();
                    		

                    }
                    break;
                case 2 :
                    // InternalBTree.g:2644:3: this_INT_1= RULE_INT
                    {
                    this_INT_1=(Token)match(input,RULE_INT,FOLLOW_2); 

                    			current.merge(this_INT_1);
                    		

                    			newLeafNode(this_INT_1, grammarAccess.getNUMBERAccess().getINTTerminalRuleCall_1());
                    		

                    }
                    break;

            }


            }


            	leaveRule();

        }

            catch (RecognitionException re) {
                recover(input,re);
                appendSkippedTokens();
            }
        finally {
        }
        return current;
    }
    // $ANTLR end "ruleNUMBER"

    // Delegated rules


 

    public static final BitSet FOLLOW_1 = new BitSet(new long[]{0x0000000000000000L});
    public static final BitSet FOLLOW_2 = new BitSet(new long[]{0x0000000000000002L});
    public static final BitSet FOLLOW_3 = new BitSet(new long[]{0x0000000000000010L});
    public static final BitSet FOLLOW_4 = new BitSet(new long[]{0x0000000000002000L});
    public static final BitSet FOLLOW_5 = new BitSet(new long[]{0x0000007A5C604000L});
    public static final BitSet FOLLOW_6 = new BitSet(new long[]{0x0000007A5C404000L});
    public static final BitSet FOLLOW_7 = new BitSet(new long[]{0x0000007A5C004000L});
    public static final BitSet FOLLOW_8 = new BitSet(new long[]{0x0000007A58004000L});
    public static final BitSet FOLLOW_9 = new BitSet(new long[]{0x0000007A50004000L});
    public static final BitSet FOLLOW_10 = new BitSet(new long[]{0x0000007A40004000L});
    public static final BitSet FOLLOW_11 = new BitSet(new long[]{0x0000007A00004000L});
    public static final BitSet FOLLOW_12 = new BitSet(new long[]{0x0000007800004000L});
    public static final BitSet FOLLOW_13 = new BitSet(new long[]{0x0000000000008000L});
    public static final BitSet FOLLOW_14 = new BitSet(new long[]{0x0000000000010000L});
    public static final BitSet FOLLOW_15 = new BitSet(new long[]{0x0000000000020000L});
    public static final BitSet FOLLOW_16 = new BitSet(new long[]{0x0040000000000020L});
    public static final BitSet FOLLOW_17 = new BitSet(new long[]{0x0000000000040000L});
    public static final BitSet FOLLOW_18 = new BitSet(new long[]{0x0000000000080000L});
    public static final BitSet FOLLOW_19 = new BitSet(new long[]{0x0000000000100000L});
    public static final BitSet FOLLOW_20 = new BitSet(new long[]{0x001E720000000000L});
    public static final BitSet FOLLOW_21 = new BitSet(new long[]{0x0000000000800010L});
    public static final BitSet FOLLOW_22 = new BitSet(new long[]{0x0000000000002002L});
    public static final BitSet FOLLOW_23 = new BitSet(new long[]{0x0000000001000010L});
    public static final BitSet FOLLOW_24 = new BitSet(new long[]{0x0000000002000020L});
    public static final BitSet FOLLOW_25 = new BitSet(new long[]{0x0000000002000000L});
    public static final BitSet FOLLOW_26 = new BitSet(new long[]{0x0000000000000040L});
    public static final BitSet FOLLOW_27 = new BitSet(new long[]{0x0000000000022000L});
    public static final BitSet FOLLOW_28 = new BitSet(new long[]{0x00400000000000E0L});
    public static final BitSet FOLLOW_29 = new BitSet(new long[]{0x00400000010000E0L});
    public static final BitSet FOLLOW_30 = new BitSet(new long[]{0x0000000002040000L});
    public static final BitSet FOLLOW_31 = new BitSet(new long[]{0x0000000080000000L});
    public static final BitSet FOLLOW_32 = new BitSet(new long[]{0x0000000128800000L});
    public static final BitSet FOLLOW_33 = new BitSet(new long[]{0x0000000120800000L});
    public static final BitSet FOLLOW_34 = new BitSet(new long[]{0x0000000000800000L});
    public static final BitSet FOLLOW_35 = new BitSet(new long[]{0x0000000400000000L});
    public static final BitSet FOLLOW_36 = new BitSet(new long[]{0x0000018128800000L});
    public static final BitSet FOLLOW_37 = new BitSet(new long[]{0x0000000000042000L});
    public static final BitSet FOLLOW_38 = new BitSet(new long[]{0x0000010128800000L});
    public static final BitSet FOLLOW_39 = new BitSet(new long[]{0x0000000000802000L});
    public static final BitSet FOLLOW_40 = new BitSet(new long[]{0x0000040000008000L});
    public static final BitSet FOLLOW_41 = new BitSet(new long[]{0x0020003800000000L});
    public static final BitSet FOLLOW_42 = new BitSet(new long[]{0x0000040000000000L});
    public static final BitSet FOLLOW_43 = new BitSet(new long[]{0x001E7A0000000000L});
    public static final BitSet FOLLOW_44 = new BitSet(new long[]{0x0000800000000000L});
    public static final BitSet FOLLOW_45 = new BitSet(new long[]{0x0001000000040000L});
    public static final BitSet FOLLOW_46 = new BitSet(new long[]{0x0000080000000000L});
    public static final BitSet FOLLOW_47 = new BitSet(new long[]{0x0000000000040002L});
    public static final BitSet FOLLOW_48 = new BitSet(new long[]{0x0000000000000020L});
    public static final BitSet FOLLOW_49 = new BitSet(new long[]{0x0080000000000000L});

}