/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree.impl;

import edu.vanderbilt.isis.alc.btree.bTree.Arg;
import edu.vanderbilt.isis.alc.btree.bTree.BBNode;
import edu.vanderbilt.isis.alc.btree.bTree.BBVar;
import edu.vanderbilt.isis.alc.btree.bTree.BTreePackage;
import edu.vanderbilt.isis.alc.btree.bTree.Topic;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>BB Node</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.BBNodeImpl#getName <em>Name</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.BBNodeImpl#getInput_topic <em>Input topic</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.BBNodeImpl#getTopic_bbvar <em>Topic bbvar</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.BBNodeImpl#getBb_vars <em>Bb vars</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.BBNodeImpl#getArgs <em>Args</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.BBNodeImpl#getComment <em>Comment</em>}</li>
 * </ul>
 *
 * @generated
 */
public class BBNodeImpl extends MinimalEObjectImpl.Container implements BBNode
{
  /**
   * The default value of the '{@link #getName() <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getName()
   * @generated
   * @ordered
   */
  protected static final String NAME_EDEFAULT = null;

  /**
   * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getName()
   * @generated
   * @ordered
   */
  protected String name = NAME_EDEFAULT;

  /**
   * The cached value of the '{@link #getInput_topic() <em>Input topic</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getInput_topic()
   * @generated
   * @ordered
   */
  protected Topic input_topic;

  /**
   * The cached value of the '{@link #getTopic_bbvar() <em>Topic bbvar</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getTopic_bbvar()
   * @generated
   * @ordered
   */
  protected BBVar topic_bbvar;

  /**
   * The cached value of the '{@link #getBb_vars() <em>Bb vars</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getBb_vars()
   * @generated
   * @ordered
   */
  protected EList<BBVar> bb_vars;

  /**
   * The cached value of the '{@link #getArgs() <em>Args</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getArgs()
   * @generated
   * @ordered
   */
  protected EList<Arg> args;

  /**
   * The default value of the '{@link #getComment() <em>Comment</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getComment()
   * @generated
   * @ordered
   */
  protected static final String COMMENT_EDEFAULT = null;

  /**
   * The cached value of the '{@link #getComment() <em>Comment</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getComment()
   * @generated
   * @ordered
   */
  protected String comment = COMMENT_EDEFAULT;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected BBNodeImpl()
  {
    super();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  protected EClass eStaticClass()
  {
    return BTreePackage.Literals.BB_NODE;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String getName()
  {
    return name;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void setName(String newName)
  {
    String oldName = name;
    name = newName;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, BTreePackage.BB_NODE__NAME, oldName, name));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public Topic getInput_topic()
  {
    if (input_topic != null && input_topic.eIsProxy())
    {
      InternalEObject oldInput_topic = (InternalEObject)input_topic;
      input_topic = (Topic)eResolveProxy(oldInput_topic);
      if (input_topic != oldInput_topic)
      {
        if (eNotificationRequired())
          eNotify(new ENotificationImpl(this, Notification.RESOLVE, BTreePackage.BB_NODE__INPUT_TOPIC, oldInput_topic, input_topic));
      }
    }
    return input_topic;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Topic basicGetInput_topic()
  {
    return input_topic;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void setInput_topic(Topic newInput_topic)
  {
    Topic oldInput_topic = input_topic;
    input_topic = newInput_topic;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, BTreePackage.BB_NODE__INPUT_TOPIC, oldInput_topic, input_topic));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public BBVar getTopic_bbvar()
  {
    if (topic_bbvar != null && topic_bbvar.eIsProxy())
    {
      InternalEObject oldTopic_bbvar = (InternalEObject)topic_bbvar;
      topic_bbvar = (BBVar)eResolveProxy(oldTopic_bbvar);
      if (topic_bbvar != oldTopic_bbvar)
      {
        if (eNotificationRequired())
          eNotify(new ENotificationImpl(this, Notification.RESOLVE, BTreePackage.BB_NODE__TOPIC_BBVAR, oldTopic_bbvar, topic_bbvar));
      }
    }
    return topic_bbvar;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public BBVar basicGetTopic_bbvar()
  {
    return topic_bbvar;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void setTopic_bbvar(BBVar newTopic_bbvar)
  {
    BBVar oldTopic_bbvar = topic_bbvar;
    topic_bbvar = newTopic_bbvar;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, BTreePackage.BB_NODE__TOPIC_BBVAR, oldTopic_bbvar, topic_bbvar));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public EList<BBVar> getBb_vars()
  {
    if (bb_vars == null)
    {
      bb_vars = new EObjectContainmentEList<BBVar>(BBVar.class, this, BTreePackage.BB_NODE__BB_VARS);
    }
    return bb_vars;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public EList<Arg> getArgs()
  {
    if (args == null)
    {
      args = new EObjectContainmentEList<Arg>(Arg.class, this, BTreePackage.BB_NODE__ARGS);
    }
    return args;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String getComment()
  {
    return comment;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void setComment(String newComment)
  {
    String oldComment = comment;
    comment = newComment;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, BTreePackage.BB_NODE__COMMENT, oldComment, comment));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs)
  {
    switch (featureID)
    {
      case BTreePackage.BB_NODE__BB_VARS:
        return ((InternalEList<?>)getBb_vars()).basicRemove(otherEnd, msgs);
      case BTreePackage.BB_NODE__ARGS:
        return ((InternalEList<?>)getArgs()).basicRemove(otherEnd, msgs);
    }
    return super.eInverseRemove(otherEnd, featureID, msgs);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public Object eGet(int featureID, boolean resolve, boolean coreType)
  {
    switch (featureID)
    {
      case BTreePackage.BB_NODE__NAME:
        return getName();
      case BTreePackage.BB_NODE__INPUT_TOPIC:
        if (resolve) return getInput_topic();
        return basicGetInput_topic();
      case BTreePackage.BB_NODE__TOPIC_BBVAR:
        if (resolve) return getTopic_bbvar();
        return basicGetTopic_bbvar();
      case BTreePackage.BB_NODE__BB_VARS:
        return getBb_vars();
      case BTreePackage.BB_NODE__ARGS:
        return getArgs();
      case BTreePackage.BB_NODE__COMMENT:
        return getComment();
    }
    return super.eGet(featureID, resolve, coreType);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @SuppressWarnings("unchecked")
  @Override
  public void eSet(int featureID, Object newValue)
  {
    switch (featureID)
    {
      case BTreePackage.BB_NODE__NAME:
        setName((String)newValue);
        return;
      case BTreePackage.BB_NODE__INPUT_TOPIC:
        setInput_topic((Topic)newValue);
        return;
      case BTreePackage.BB_NODE__TOPIC_BBVAR:
        setTopic_bbvar((BBVar)newValue);
        return;
      case BTreePackage.BB_NODE__BB_VARS:
        getBb_vars().clear();
        getBb_vars().addAll((Collection<? extends BBVar>)newValue);
        return;
      case BTreePackage.BB_NODE__ARGS:
        getArgs().clear();
        getArgs().addAll((Collection<? extends Arg>)newValue);
        return;
      case BTreePackage.BB_NODE__COMMENT:
        setComment((String)newValue);
        return;
    }
    super.eSet(featureID, newValue);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eUnset(int featureID)
  {
    switch (featureID)
    {
      case BTreePackage.BB_NODE__NAME:
        setName(NAME_EDEFAULT);
        return;
      case BTreePackage.BB_NODE__INPUT_TOPIC:
        setInput_topic((Topic)null);
        return;
      case BTreePackage.BB_NODE__TOPIC_BBVAR:
        setTopic_bbvar((BBVar)null);
        return;
      case BTreePackage.BB_NODE__BB_VARS:
        getBb_vars().clear();
        return;
      case BTreePackage.BB_NODE__ARGS:
        getArgs().clear();
        return;
      case BTreePackage.BB_NODE__COMMENT:
        setComment(COMMENT_EDEFAULT);
        return;
    }
    super.eUnset(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public boolean eIsSet(int featureID)
  {
    switch (featureID)
    {
      case BTreePackage.BB_NODE__NAME:
        return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
      case BTreePackage.BB_NODE__INPUT_TOPIC:
        return input_topic != null;
      case BTreePackage.BB_NODE__TOPIC_BBVAR:
        return topic_bbvar != null;
      case BTreePackage.BB_NODE__BB_VARS:
        return bb_vars != null && !bb_vars.isEmpty();
      case BTreePackage.BB_NODE__ARGS:
        return args != null && !args.isEmpty();
      case BTreePackage.BB_NODE__COMMENT:
        return COMMENT_EDEFAULT == null ? comment != null : !COMMENT_EDEFAULT.equals(comment);
    }
    return super.eIsSet(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String toString()
  {
    if (eIsProxy()) return super.toString();

    StringBuilder result = new StringBuilder(super.toString());
    result.append(" (name: ");
    result.append(name);
    result.append(", comment: ");
    result.append(comment);
    result.append(')');
    return result.toString();
  }

} //BBNodeImpl