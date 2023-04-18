/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree.impl;

import edu.vanderbilt.isis.alc.btree.bTree.BTreePackage;
import edu.vanderbilt.isis.alc.btree.bTree.ChildNode;
import edu.vanderbilt.isis.alc.btree.bTree.ParBTNode;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Par BT Node</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.ParBTNodeImpl#getName <em>Name</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.ParBTNodeImpl#getCond <em>Cond</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.ParBTNodeImpl#getNodes <em>Nodes</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ParBTNodeImpl extends BTreeNodeImpl implements ParBTNode
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
   * The default value of the '{@link #getCond() <em>Cond</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getCond()
   * @generated
   * @ordered
   */
  protected static final String COND_EDEFAULT = null;

  /**
   * The cached value of the '{@link #getCond() <em>Cond</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getCond()
   * @generated
   * @ordered
   */
  protected String cond = COND_EDEFAULT;

  /**
   * The cached value of the '{@link #getNodes() <em>Nodes</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getNodes()
   * @generated
   * @ordered
   */
  protected EList<ChildNode> nodes;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected ParBTNodeImpl()
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
    return BTreePackage.Literals.PAR_BT_NODE;
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
      eNotify(new ENotificationImpl(this, Notification.SET, BTreePackage.PAR_BT_NODE__NAME, oldName, name));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String getCond()
  {
    return cond;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void setCond(String newCond)
  {
    String oldCond = cond;
    cond = newCond;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, BTreePackage.PAR_BT_NODE__COND, oldCond, cond));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public EList<ChildNode> getNodes()
  {
    if (nodes == null)
    {
      nodes = new EObjectContainmentEList<ChildNode>(ChildNode.class, this, BTreePackage.PAR_BT_NODE__NODES);
    }
    return nodes;
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
      case BTreePackage.PAR_BT_NODE__NODES:
        return ((InternalEList<?>)getNodes()).basicRemove(otherEnd, msgs);
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
      case BTreePackage.PAR_BT_NODE__NAME:
        return getName();
      case BTreePackage.PAR_BT_NODE__COND:
        return getCond();
      case BTreePackage.PAR_BT_NODE__NODES:
        return getNodes();
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
      case BTreePackage.PAR_BT_NODE__NAME:
        setName((String)newValue);
        return;
      case BTreePackage.PAR_BT_NODE__COND:
        setCond((String)newValue);
        return;
      case BTreePackage.PAR_BT_NODE__NODES:
        getNodes().clear();
        getNodes().addAll((Collection<? extends ChildNode>)newValue);
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
      case BTreePackage.PAR_BT_NODE__NAME:
        setName(NAME_EDEFAULT);
        return;
      case BTreePackage.PAR_BT_NODE__COND:
        setCond(COND_EDEFAULT);
        return;
      case BTreePackage.PAR_BT_NODE__NODES:
        getNodes().clear();
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
      case BTreePackage.PAR_BT_NODE__NAME:
        return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
      case BTreePackage.PAR_BT_NODE__COND:
        return COND_EDEFAULT == null ? cond != null : !COND_EDEFAULT.equals(cond);
      case BTreePackage.PAR_BT_NODE__NODES:
        return nodes != null && !nodes.isEmpty();
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
    result.append(", cond: ");
    result.append(cond);
    result.append(')');
    return result.toString();
  }

} //ParBTNodeImpl