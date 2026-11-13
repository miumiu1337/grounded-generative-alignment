"""Validate quoted edit spans and conservative Variant C decisions."""

CORRECTION={'type':'object','properties':{'old_text':{'type':'string','minLength':1,'maxLength':300},
    'new_text':{'type':'string','maxLength':300},'reference_id':{'type':'string'},
    'quote':{'type':'string','minLength':1,'maxLength':500}},
    'required':['old_text','new_text','reference_id','quote'],'additionalProperties':False}
PROPOSAL_SCHEMA={'type':'object','properties':{'translation':{'type':'string','minLength':1},
    'corrections':{'type':'array','maxItems':3,'items':CORRECTION}},
    'required':['translation','corrections'],'additionalProperties':False}
VERIFICATION_SCHEMA={'type':'object','properties':{'supported':{'type':'boolean'},'faithful':{'type':'boolean'},
    'complete':{'type':'boolean'},'reason':{'type':'string','minLength':1,'maxLength':500}},
    'required':['supported','faithful','complete','reason'],'additionalProperties':False}


def validate_proposal(value,blind,sources):
    if (not isinstance(value,dict) or set(value)!={'translation','corrections'}
            or not isinstance(value['translation'],str) or not value['translation'].strip()
            or not isinstance(value['corrections'],list) or len(value['corrections'])>3):
        raise ValueError('Malformed C correction proposal')
    references={s['reference_id']:s for s in sources};text=blind
    for edit in value['corrections']:
        if not isinstance(edit,dict) or set(edit)!={'old_text','new_text','reference_id','quote'}:
            raise ValueError('Malformed C correction')
        if any(not isinstance(edit[k],str) for k in edit):raise ValueError('C correction fields must be strings')
        old,new,ref,quote=(edit[k] for k in ('old_text','new_text','reference_id','quote'))
        if not old or old==new or len(old)>300 or len(new)>300 or text.count(old)!=1:
            raise ValueError('C correction must identify one unambiguous changed span')
        if ref not in references or not quote.strip() or len(quote)>500 or quote not in references[ref]['text']:
            raise ValueError('C correction lacks an exact quote from its declared reference')
        text=text.replace(old,new,1)
    if text!=value['translation']:raise ValueError('C translation contains edits outside the supported correction list')
    return value


def validate_verification(value):
    if (not isinstance(value,dict) or set(value)!={'supported','faithful','complete','reason'}
            or any(type(value[k]) is not bool for k in ('supported','faithful','complete'))
            or not isinstance(value['reason'],str) or not value['reason'].strip() or len(value['reason'])>500):
        raise ValueError('Malformed C verification')
    return value


def accept(verification):
    validate_verification(verification)
    return all(verification[k] for k in ('supported','faithful','complete'))
